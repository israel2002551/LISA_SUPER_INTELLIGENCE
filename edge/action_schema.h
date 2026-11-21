#pragma once
#include <ctype.h>
// ==============================================================================
// Device profiles, prompt format, and action validation
// ==============================================================================
// SAFETY: the model never writes to hardware directly. Its output must be "NOP"
// or a JSON action naming a command the peripheral declared in its IDENTIFY
// manifest, with every argument matching the declared spec. Anything else is
// logged as invalid_output (training signal for the server) and nothing is sent.
//
// Arg specs (same grammar in server/app/evolution/action_schema.py):
//   "int" | "int:MIN..MAX" | "float" | "float:MIN..MAX" | "bool"
//   "str" | "str:MAXLEN"   | "enum:a|b|c"
// ==============================================================================

#include <Arduino.h>
#include <ArduinoJson.h>
#include <stdlib.h>
#include "edge_config.h"

#define EDGE_MAX_COMMANDS 8
#define EDGE_MAX_ARGS     4

struct ArgSpec {
    char name[16];
    char spec[40];
};

struct CommandSpec {
    char cmd[24];
    ArgSpec args[EDGE_MAX_ARGS];
    uint8_t n_args;
};

struct DeviceProfile {
    bool present = false;          // something is talking on the port
    bool identified = false;       // answered IDENTIFY with a manifest
    bool manifest_truncated = false;
    char key[40] = "";             // manifest id, or "raw:uart1" for unidentified devices
    char type[24] = "";            // manifest type, or cloud-classified intent
    char format[16] = "unknown";   // json_lines | nmea | ascii_lines | csv | modbus_rtu | binary | unknown
    char link[12] = "uart";        // uart | uart_fast | espnow | tcp | spi
    uint32_t baud = 0;
    uint32_t max_baud = 0;
    bool acks_actions = true;      // peripheral replies {"ack":<id>,"ok":bool}
    char links[48] = "";           // manifest bus.links, comma-separated
    CommandSpec accepts[EDGE_MAX_COMMANDS];
    uint8_t n_accepts = 0;
    char manifest[512] = "";       // raw manifest JSON for telemetry
};

enum ActionVerdict { ACTION_OK, ACTION_NOP, ACTION_INVALID };

// A shared blank profile. DeviceProfile is ~2.7 KB; resetting with
// `p = DeviceProfile()` would build that temporary on a small task stack.
static const DeviceProfile& empty_profile() {
    static const DeviceProfile empty{};
    return empty;
}

// ---- Manifest -------------------------------------------------------------------
// {"id":"thermo-01","type":"thermostat","fw":"1.0","emits":["temp"],
//  "accepts":[{"cmd":"set_fan","args":{"level":"int:0..3"}}],
//  "needs":["alerts"],"acks":true,"bus":{"max_baud":2000000,"links":["uart"]}}
static bool profile_from_manifest(const char* line, uint32_t baud, DeviceProfile& p) {
    JsonDocument doc;
    if (deserializeJson(doc, line) != DeserializationError::Ok) return false;
    const char* id = doc["id"] | "";
    if (!*id) return false;

    p = empty_profile();
    p.present = true;
    p.identified = true;
    p.baud = baud;
    strlcpy(p.key, id, sizeof(p.key));
    strlcpy(p.type, doc["type"] | "device", sizeof(p.type));
    strlcpy(p.format, "json_lines", sizeof(p.format));
    p.max_baud = doc["bus"]["max_baud"] | 0;
    p.acks_actions = doc["acks"] | true;

    for (JsonVariantConst l : doc["bus"]["links"].as<JsonArrayConst>()) {
        const char* name = l | "";
        if (strlen(p.links) + strlen(name) + 2 < sizeof(p.links)) {
            if (p.links[0]) strcat(p.links, ",");
            strcat(p.links, name);
        }
    }

    for (JsonObjectConst c : doc["accepts"].as<JsonArrayConst>()) {
        if (p.n_accepts >= EDGE_MAX_COMMANDS) break;
        CommandSpec& cs = p.accepts[p.n_accepts];
        strlcpy(cs.cmd, c["cmd"] | "", sizeof(cs.cmd));
        if (!cs.cmd[0]) continue;
        cs.n_args = 0;
        for (JsonPairConst kv : c["args"].as<JsonObjectConst>()) {
            if (cs.n_args >= EDGE_MAX_ARGS) break;
            strlcpy(cs.args[cs.n_args].name, kv.key().c_str(), sizeof(cs.args[0].name));
            strlcpy(cs.args[cs.n_args].spec, kv.value() | "str", sizeof(cs.args[0].spec));
            cs.n_args++;
        }
        p.n_accepts++;
    }

    size_t len = strlen(line);
    p.manifest_truncated = len >= sizeof(p.manifest);
    strlcpy(p.manifest, line, sizeof(p.manifest));
    return true;
}

// ---- Prompt (contract with server/app/evolution/prompt_format.py) ----------------
static int edge_build_prompt(const DeviceProfile& p, const char* event, char* out, size_t out_len) {
    char ev[EDGE_EVENT_MAX_CHARS + 1];
    size_t n = 0;
    for (; event[n] && n < EDGE_EVENT_MAX_CHARS; n++) {
        char c = event[n];
        ev[n] = (c == '\r' || c == '\n') ? ' ' : c;
    }
    ev[n] = '\0';
    return snprintf(out, out_len, "<d>%s:%s<e>%s<a>", p.key, p.type, ev);
}

// Chat (contract with build_chat_prompt in server/app/evolution/prompt_format.py):
//   "<u>" + text + "<l>"                    or, with memory,
//   "<l>" + prev + "<u>" + text + "<l>"     for a follow-up, if <= EDGE_CHAT_HISTORY_MAX_BYTES.
// Each part: CR/LF -> space, ends trimmed, cut to EDGE_CHAT_MAX_CHARS bytes.
static size_t edge_chat_text(const char* text, char* t) {   // t: EDGE_CHAT_MAX_CHARS + 1 bytes
    auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    const char* s = text ? text : "";
    while (blank(*s)) s++;
    size_t len = strlen(s);
    while (len && blank(s[len - 1])) len--;
    if (len > EDGE_CHAT_MAX_CHARS) len = EDGE_CHAT_MAX_CHARS;
    for (size_t i = 0; i < len; i++) t[i] = (s[i] == '\r' || s[i] == '\n') ? ' ' : s[i];
    t[len] = '\0';
    return len;
}

// Follow-ups: the only messages that get her previous reply as context (FOLLOWUPS in
// prompt_format.py, same order). Given context on every message, the 2.75M model
// started copying her last reply back to ordinary questions.
static const char* const kChatFollowups[] = {
    "say that again", "repeat that", "what did you say", "can you repeat that", "pardon",
    "come again", "say it again", "repeat please",
    "tell me more", "why", "what do you mean", "explain", "go on", "how so",
    "can you explain that", "tell me more about it",
};

// normalize_message(): lowercase, trimmed, end punctuation "?!. " removed, a leading
// "lisa, "/"lisa " and trailing ", lisa"/" lisa" removed, whitespace runs -> one space.
// Returns false if the text is too long to be a follow-up.
static bool edge_normalize_message(const char* text, char* out, size_t out_len) {
    auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    char t[64];
    const char* s = text;
    while (blank(*s)) s++;
    size_t len = strlen(s);
    while (len && blank(s[len - 1])) len--;
    if (len >= sizeof(t)) return false;
    for (size_t i = 0; i < len; i++) t[i] = (char)tolower((unsigned char)s[i]);
    t[len] = '\0';
    while (len && strchr("?!. ", t[len - 1])) t[--len] = '\0';
    char* p = t;
    if (strncmp(p, "lisa, ", 6) == 0) p += 6;
    if (strncmp(p, "lisa ", 5) == 0) p += 5;
    len = strlen(p);
    if (len >= 6 && strcmp(p + len - 6, ", lisa") == 0) p[len -= 6] = '\0';
    if (len >= 5 && strcmp(p + len - 5, " lisa") == 0) p[len -= 5] = '\0';
    size_t n = 0;
    bool gap = false;
    for (; *p && n + 2 < out_len; p++) {
        if (blank(*p)) { gap = n > 0; continue; }
        if (gap) out[n++] = ' ';
        gap = false;
        out[n++] = *p;
    }
    out[n] = '\0';
    return true;
}

static bool edge_is_followup(const char* text) {
    char m[64];
    if (!edge_normalize_message(text, m, sizeof(m))) return false;
    for (const char* f : kChatFollowups) {
        if (strcmp(m, f) == 0) return true;
    }
    return false;
}

static int edge_build_chat_prompt(const char* text, const char* prev, char* out, size_t out_len) {
    char t[EDGE_CHAT_MAX_CHARS + 1], p[EDGE_CHAT_MAX_CHARS + 1];
    size_t tl = edge_chat_text(text, t);
    size_t pl = (prev && edge_is_followup(t)) ? edge_chat_text(prev, p) : 0;
    if (pl && 9 + pl + tl <= EDGE_CHAT_HISTORY_MAX_BYTES) return snprintf(out, out_len, "<l>%s<u>%s<l>", p, t);
    return snprintf(out, out_len, "<u>%s<l>", t);
}

// ---- Validation -------------------------------------------------------------------
// "LO..HI", e.g. "0..3", "-1.5..2.5". Split on ".." first: strtod("0..3") would
// consume "0." as a number and leave ".3", so every integer range used to fail
// (every action with a ranged arg was rejected as invalid).
static bool parse_range(const char* s, double& lo, double& hi) {
    const char* dots = strstr(s, "..");
    if (!dots || dots == s) return false;
    char first[24];
    size_t n = (size_t)(dots - s);
    if (n >= sizeof(first)) return false;
    memcpy(first, s, n);
    first[n] = '\0';
    char* end;
    lo = strtod(first, &end);
    if (end == first || *end != '\0') return false;
    const char* second = dots + 2;
    hi = strtod(second, &end);
    return end != second && *end == '\0';
}

static bool arg_matches(const char* spec, JsonVariantConst v) {
    const char* colon = strchr(spec, ':');
    size_t kind_len = colon ? (size_t)(colon - spec) : strlen(spec);
    const char* param = colon ? colon + 1 : nullptr;
    auto kind_is = [&](const char* k) { return strlen(k) == kind_len && strncmp(spec, k, kind_len) == 0; };

    if (kind_is("int")) {
        if (!v.is<long>() || v.is<bool>()) return false;
        if (!param) return true;
        double lo, hi;
        long x = v.as<long>();
        return parse_range(param, lo, hi) && x >= lo && x <= hi;
    }
    if (kind_is("float")) {
        if (!(v.is<float>() || v.is<long>()) || v.is<bool>()) return false;
        if (!param) return true;
        double lo, hi;
        double x = v.as<double>();
        return parse_range(param, lo, hi) && x >= lo && x <= hi;
    }
    if (kind_is("bool")) return v.is<bool>();
    if (kind_is("str")) {
        if (!v.is<const char*>()) return false;
        return !param || strlen(v.as<const char*>()) <= (size_t)atoi(param);
    }
    if (kind_is("enum")) {
        if (!v.is<const char*>() || !param) return false;
        const char* val = v.as<const char*>();
        size_t vl = strlen(val);
        const char* p = param;
        while (*p) {
            const char* bar = strchr(p, '|');
            size_t opt_len = bar ? (size_t)(bar - p) : strlen(p);
            if (opt_len == vl && strncmp(p, val, vl) == 0) return true;
            if (!bar) break;
            p = bar + 1;
        }
        return false;
    }
    return false;   // unknown spec kind: reject
}

// Checks model output against the device's manifest. On ACTION_OK, `action`
// holds {"cmd":...,"args":{...}}. `why` explains an ACTION_INVALID.
static ActionVerdict validate_action(const DeviceProfile& p, const char* text, JsonDocument& action, char* why, size_t why_len) {
    while (*text == ' ') text++;
    if (strncmp(text, "NOP", 3) == 0) return ACTION_NOP;

    action.clear();
    if (deserializeJson(action, text) != DeserializationError::Ok || !action.is<JsonObject>()) {
        snprintf(why, why_len, "not JSON");
        return ACTION_INVALID;
    }
    const char* cmd = action["cmd"] | "";
    const CommandSpec* cs = nullptr;
    for (int i = 0; i < p.n_accepts; i++) {
        if (strcmp(p.accepts[i].cmd, cmd) == 0) cs = &p.accepts[i];
    }
    if (!cs) {
        snprintf(why, why_len, "unknown cmd '%s'", cmd);
        return ACTION_INVALID;
    }

    JsonObjectConst args = action["args"].as<JsonObjectConst>();
    for (JsonPairConst kv : args) {
        const ArgSpec* as = nullptr;
        for (int i = 0; i < cs->n_args; i++) {
            if (strcmp(cs->args[i].name, kv.key().c_str()) == 0) as = &cs->args[i];
        }
        if (!as) {
            snprintf(why, why_len, "unknown arg '%s'", kv.key().c_str());
            return ACTION_INVALID;
        }
        if (!arg_matches(as->spec, kv.value())) {
            snprintf(why, why_len, "arg '%s' violates %s", as->name, as->spec);
            return ACTION_INVALID;
        }
    }
    for (int i = 0; i < cs->n_args; i++) {
        if (args.isNull() || args[cs->args[i].name].isNull()) {
            snprintf(why, why_len, "missing arg '%s'", cs->args[i].name);
            return ACTION_INVALID;
        }
    }
    return ACTION_OK;
}
