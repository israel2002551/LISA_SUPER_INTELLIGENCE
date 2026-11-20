#pragma once
// ==============================================================================
// Edge node secrets: TEMPLATE
// ==============================================================================
// Copy this file to edge/secrets.h (git-ignored) and fill it in. Without
// secrets.h the firmware still builds and runs offline (UART + local inference +
// ring-buffer logging), but it can't reach the evolution server.

#define EDGE_WIFI_SSID          ""
#define EDGE_WIFI_PASSWORD      ""

// Must match a row in public.edge_nodes (create it with server/scripts/register_node.py,
// which also prints the API key below; the server stores only its SHA-256).
#define EDGE_NODE_ID            "edge-01"
#define EDGE_NODE_API_KEY       ""

// Evolution server, no trailing slash. e.g. "https://your-service.onrender.com"
#define EDGE_SERVER_BASE_URL    ""

// PEM root certificate(s) that sign the server's and Supabase Storage's TLS
// certificates. Several PEM blocks can be concatenated. See docs/BRINGUP.md for
// how to fetch them with openssl.
#define EDGE_TLS_ROOT_CA        ""

// ECDSA P-256 public key that verifies OTA manifests. Print it with:
//   python server/scripts/gen_signing_key.py --show-public
#define EDGE_OTA_PUBLIC_KEY_PEM ""

// ---- Development-only escape hatches. Leave these undefined in production. ----
// #define EDGE_TLS_INSECURE         // skip TLS certificate checks
// #define EDGE_ALLOW_PLAIN_HTTP     // allow an http:// server (LAN testing against a laptop)
// #define EDGE_ALLOW_UNSIGNED_OTA   // accept OTA manifests without a valid signature
