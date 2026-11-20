-- =============================================================================
-- LISA (self-evolving edge AI): core schema
-- =============================================================================
-- Access model: only the evolution server touches these tables, using the
-- service_role key (which bypasses RLS). RLS is enabled with NO policies, so
-- the anon/authenticated keys can read or write nothing. Devices never talk to
-- Supabase directly except to download model files through short-lived signed
-- URLs the server creates.
-- =============================================================================

create extension if not exists pgcrypto;   -- gen_random_uuid()

-- Keeps updated_at current on UPDATE.
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

-- -----------------------------------------------------------------------------
-- evolution_runs: one row per self-improvement cycle (curate -> train -> publish)
-- -----------------------------------------------------------------------------
create table public.evolution_runs (
  id                uuid primary key default gen_random_uuid(),
  status            text not null default 'queued'
                    check (status in ('queued', 'running', 'published', 'rejected', 'failed')),
  trigger           text not null default 'manual'
                    check (trigger in ('manual', 'schedule', 'threshold')),
  trainer           text,                          -- mock | torch | remote
  params            jsonb not null default '{}'::jsonb,
  stats             jsonb not null default '{}'::jsonb,   -- curation counts, metrics, gate decision
  error             text,
  produced_version  integer,                       -- FK added after model_versions exists
  created_at        timestamptz not null default now(),
  started_at        timestamptz,
  finished_at       timestamptz
);
create index evolution_runs_status_idx on public.evolution_runs (status, created_at);

-- -----------------------------------------------------------------------------
-- model_versions: every evolved model, its storage location and OTA metadata
-- -----------------------------------------------------------------------------
create table public.model_versions (
  version              integer primary key check (version > 0),   -- written into the image header
  label                text,
  base_model           text not null default 'lisa-gemma4-int16-h256-l2-v4096',
  parent_version       integer references public.model_versions (version) on delete set null,
  quantization_format  text not null default 'q16_rowscale_tied',
  size_bytes           bigint not null check (size_bytes > 0),
  sha256               text not null unique check (sha256 ~ '^[0-9a-f]{64}$'),
  signature            text,              -- base64 DER ECDSA P-256 over 'W25OTA1|version|size|sha256'
  storage_path         text not null,     -- object path in the 'models' bucket
  delta_from_version   integer references public.model_versions (version) on delete set null,
  delta_storage_path   text,              -- object path in the 'deltas' bucket
  delta_size_bytes     bigint,
  delta_sha256         text check (delta_sha256 is null or delta_sha256 ~ '^[0-9a-f]{64}$'),
  metrics_json         jsonb not null default '{}'::jsonb,
  training_run_id      uuid references public.evolution_runs (id) on delete set null,
  is_active            boolean not null default false,
  created_at           timestamptz not null default now(),
  check ((delta_from_version is null) = (delta_storage_path is null))
);
-- At most one active model (the one devices are told to install).
create unique index model_versions_one_active on public.model_versions (is_active) where is_active;

alter table public.evolution_runs
  add constraint evolution_runs_produced_version_fkey
  foreign key (produced_version) references public.model_versions (version) on delete set null;

-- -----------------------------------------------------------------------------
-- edge_nodes: registered ESP32-S3 devices
-- -----------------------------------------------------------------------------
create table public.edge_nodes (
  id                    text primary key check (id ~ '^[A-Za-z0-9_-]{1,40}$'),   -- EDGE_NODE_ID
  display_name          text,
  api_key_hash          text not null check (api_key_hash ~ '^[0-9a-f]{64}$'),  -- sha256 of the node key
  hw_model              text not null default 'esp32-s3',
  fw_version            text,
  w25q_jedec_id         text,
  w25q_total_bytes      bigint,
  active_slot           smallint check (active_slot in (0, 1)),
  active_model_version  integer references public.model_versions (version) on delete set null,
  active_model_sha256   text,
  partition_state       jsonb not null default '{}'::jsonb,   -- slots, ring-log cursors (from telemetry)
  status                jsonb not null default '{}'::jsonb,   -- last full status report
  last_heartbeat_at     timestamptz,
  created_at            timestamptz not null default now(),
  updated_at            timestamptz not null default now()
);
create trigger edge_nodes_touch before update on public.edge_nodes
  for each row execute function public.touch_updated_at();

-- -----------------------------------------------------------------------------
-- uart_devices: peripherals discovered on a node's UART ports
-- -----------------------------------------------------------------------------
create table public.uart_devices (
  id                   uuid primary key default gen_random_uuid(),
  node_id              text not null references public.edge_nodes (id) on delete cascade,
  device_key           text not null,     -- manifest id, or 'raw:uart1' for unidentified devices
  port                 smallint not null default 1,
  identified           boolean not null default false,
  baud                 integer,
  bus                  text not null default 'uart'
                       check (bus in ('uart', 'uart_fast', 'espnow', 'tcp', 'spi')),
  data_format          text not null default 'unknown',
  device_type          text,
  capability_manifest  jsonb not null default '{}'::jsonb,   -- IDENTIFY reply
  inferred_intent      text,                                -- what the device wants from the AI
  protocol_schema      jsonb not null default '{}'::jsonb,   -- learned framing / fields (cloud classifier)
  learned_rules        jsonb not null default '[]'::jsonb,   -- [{when:{...}, do:{cmd,args}}] used by the rule teacher
  first_seen_at        timestamptz not null default now(),
  last_seen_at         timestamptz not null default now(),
  updated_at           timestamptz not null default now(),
  unique (node_id, device_key)
);
create trigger uart_devices_touch before update on public.uart_devices
  for each row execute function public.touch_updated_at();

-- -----------------------------------------------------------------------------
-- interaction_logs: every UART/system event uploaded from the ring log
-- -----------------------------------------------------------------------------
create table public.interaction_logs (
  id                 bigint generated always as identity primary key,
  node_id            text not null references public.edge_nodes (id) on delete cascade,
  uart_device_id     uuid references public.uart_devices (id) on delete set null,
  device_seq         bigint not null,     -- ring-log seq: makes re-uploads idempotent
  event_kind         text not null
                     check (event_kind in ('interaction', 'device', 'handshake', 'observation',
                                           'feedback', 'ota', 'boot', 'error')),
  act_id             bigint,              -- links an interaction to later peripheral feedback
  raw_input          text,
  prompt             text,
  model_output       text,
  action             jsonb,
  model_version      integer,
  latency_ms         integer,
  status             text check (status in ('ok', 'nop', 'invalid_output', 'rejected', 'timeout', 'error')),
  error_detail       text,
  payload            jsonb not null default '{}'::jsonb,   -- full record as uploaded
  reward             real check (reward between 0 and 1),
  used_for_training  boolean not null default false,
  received_at        timestamptz not null default now(),
  unique (node_id, device_seq)
);
-- Curator scans untrained interactions oldest-first.
create index interaction_logs_untrained_idx on public.interaction_logs (received_at)
  where used_for_training = false and event_kind = 'interaction';
create index interaction_logs_node_idx on public.interaction_logs (node_id, received_at desc);
create index interaction_logs_device_idx on public.interaction_logs (uart_device_id);
create index interaction_logs_act_idx on public.interaction_logs (node_id, act_id) where act_id is not null;

-- -----------------------------------------------------------------------------
-- training_datasets: curated + synthetic examples (prompt -> target)
-- -----------------------------------------------------------------------------
create table public.training_datasets (
  id               bigint generated always as identity primary key,
  prompt           text not null,
  context_json     jsonb not null default '{}'::jsonb,
  target_response  text not null,
  source_type      text not null check (source_type in ('real_log', 'synthetic_distill', 'failure_recovery')),
  split            text not null default 'train' check (split in ('train', 'val')),
  source_log_id    bigint references public.interaction_logs (id) on delete set null,
  uart_device_id   uuid references public.uart_devices (id) on delete set null,
  teacher          text,                  -- 'log' | 'rule' | 'claude-opus-5-5'
  quality          real check (quality between 0 and 1),
  dedupe_key       text not null unique,  -- sha256(prompt || 0x00 || target)
  created_at       timestamptz not null default now()
);
create index training_datasets_split_idx on public.training_datasets (split, source_type);

-- -----------------------------------------------------------------------------
-- Atomic activation: exactly one active model, switched in one transaction.
-- -----------------------------------------------------------------------------
create or replace function public.activate_model_version(p_version integer)
returns void
language plpgsql
set search_path = public
as $$
begin
  if not exists (select 1 from public.model_versions where version = p_version) then
    raise exception 'unknown model version %', p_version;
  end if;
  update public.model_versions set is_active = false where is_active and version <> p_version;
  update public.model_versions set is_active = true where version = p_version;
end;
$$;
revoke all on function public.activate_model_version(integer) from public, anon, authenticated;
grant execute on function public.activate_model_version(integer) to service_role;

-- -----------------------------------------------------------------------------
-- Row Level Security: on, with no policies (service_role only).
-- -----------------------------------------------------------------------------
alter table public.edge_nodes        enable row level security;
alter table public.uart_devices      enable row level security;
alter table public.interaction_logs  enable row level security;
alter table public.model_versions    enable row level security;
alter table public.training_datasets enable row level security;
alter table public.evolution_runs    enable row level security;
