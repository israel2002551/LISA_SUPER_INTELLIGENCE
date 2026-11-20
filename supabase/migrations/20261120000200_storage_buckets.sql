-- =============================================================================
-- Storage buckets for model artifacts
-- =============================================================================
--   models/  full quantized images:  v{version}.w25img        (<= 6 MiB, one W25Q slot)
--   deltas/  block deltas:           v{from}_to_v{to}.w25d
--
-- Both are private. The evolution server uploads with the service_role key and
-- hands devices short-lived signed URLs (POST /storage/v1/object/sign/...), so
-- no storage policies are needed for anon/authenticated users.
-- =============================================================================

insert into storage.buckets (id, name, public, file_size_limit)
values
  ('models', 'models', false, 16777216),   -- 16 MiB cap (a W25Q slot is 6 MiB)
  ('deltas', 'deltas', false, 16777216)
on conflict (id) do nothing;
