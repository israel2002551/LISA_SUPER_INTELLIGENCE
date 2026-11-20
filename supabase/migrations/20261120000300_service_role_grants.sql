 -- =============================================================================
-- LISA: table privileges for the API roles
-- =============================================================================
-- Newer Supabase projects don't grant the API roles access to tables created in
-- `public`, so PostgREST answered 403 / 42501 even for service_role. The LISA
-- server only ever uses service_role: grant exactly that, and make sure the
-- public roles (anon, authenticated) have nothing, on top of RLS.
-- =============================================================================

grant usage on schema public to service_role;

grant select, insert, update, delete on table
  public.edge_nodes,
  public.uart_devices,
  public.interaction_logs,
  public.training_datasets,
  public.model_versions,
  public.evolution_runs
to service_role;

-- identity columns (interaction_logs.id, training_datasets.id) draw from sequences
grant usage, select on all sequences in schema public to service_role;

grant execute on function public.touch_updated_at() to service_role;
grant execute on function public.activate_model_version(integer) to service_role;

revoke all on table
  public.edge_nodes,
  public.uart_devices,
  public.interaction_logs,
  public.training_datasets,
  public.model_versions,
  public.evolution_runs
from anon, authenticated;
