"""Author-maintained fictional troubleshooting corpus. Writes deterministic local JSONL assets."""

import json
from pathlib import Path


FAMILIES = [
    {
        "category": "DB_POOL",
        "check": "INSPECT_CONNECTION_HOLDERS",
        "reason": "Connection acquisition waits can rise even when query execution stays stable.",
        "primary": "Database pool exhaustion: inspect active connection owners, pending waiters, and long transactions before changing pool limits. A timed-out acquisition does not prove a slow SQL query.",
        "secondary": "Database capacity notes: compare connection utilization and wait time against baseline. Confirm whether clients close connections and whether transaction spans grew.",
        "tickets": [
            "Checkout requests time out waiting to borrow a database connection; query duration stays ordinary.",
            "The API's pool wait queue grows while the database reports nearly all connections occupied.",
            "Clients cannot acquire a free SQL session, but statements finish at their usual speed.",
            "Orders intermittently fail after a long connection checkout wait; active sessions are high.",
            "Connection acquisition stalls with no corresponding jump in execution latency.",
            "Requests hang before SQL starts, and the pool is at capacity.",
            "Web handlers wait for a free database handle while query runtimes remain flat.",
            "Most sessions are in use; application threads block borrowing database slots.",
        ],
    },
    {
        "category": "UPSTREAM_TIMEOUT",
        "check": "COMPARE_DEPENDENCY_LATENCY",
        "reason": "Dependency response times and timeout rates separate external delay from local work.",
        "primary": "Upstream timeout: compare the dependency's latency and timeout rate to the caller's baseline; inspect the deadline and retries. Do not blindly add retries, which may amplify load.",
        "secondary": "External service diagnosis: correlate client deadline errors with server-side response latency. Distinguish network timeouts from fast application rejections.",
        "tickets": [
            "The checkout API waits on the payment provider until the client deadline expires.",
            "Calls to a downstream gateway are slow and eventually time out, while local compute is steady.",
            "The remote billing service fails to answer before our request deadline.",
            "Requests complete locally but block at an external service response.",
            "Latency spike tracks a dependency's long response time and repeated timeouts.",
            "The external API stops replying quickly; our worker times out after waiting.",
            "Checkout stalls on a remote charge call, despite normal local database timings.",
            "An upstream provider response exceeds the configured client deadline.",
        ],
    },
    {
        "category": "RESTART_LOOP",
        "check": "INSPECT_CONTAINER_TERMINATION",
        "reason": "Termination reason and restart count distinguish crash, OOM, and failing probes.",
        "primary": "Restart loop: inspect container last termination reason, exit code, probe failures, and memory limit before modifying deployment replicas. CrashLoopBackOff is a symptom, not a root cause.",
        "secondary": "Service process startup: compare boot logs with pod restart timestamps and resource limits. Verify whether readiness or liveness probes kill the process.",
        "tickets": [
            "The workload repeatedly exits and the orchestrator keeps relaunching its container.",
            "Pods never stabilize; the restart counter rises after each short startup attempt.",
            "A service process boots, crashes, and starts again every minute.",
            "The deployment remains unavailable because its containers continually restart.",
            "Workers begin initialization but are terminated before becoming healthy.",
            "A container cycles between starting and stopped, with the restart count increasing.",
            "The application briefly starts, then the pod is relaunched over and over.",
            "Short-lived containers keep getting recreated; health never stays green.",
        ],
    },
    {
        "category": "TLS_CERT",
        "check": "INSPECT_CERTIFICATE_CHAIN",
        "reason": "Certificate validity, chain, and hostname checks distinguish handshake failures.",
        "primary": "TLS certificate failure: check expiration, server name, chain, issuer, and client trust store. Avoid disabling certificate verification in production.",
        "secondary": "Secure connection troubleshooting: inspect handshake error text and the peer certificate. Compare clock skew and recent certificate rotation.",
        "tickets": [
            "HTTPS clients reject a service certificate after a scheduled rotation.",
            "Secure requests fail during handshake because the peer identity cannot be verified.",
            "An API endpoint presents a certificate that browsers no longer trust.",
            "Traffic cannot establish an encrypted session; hostname validation fails.",
            "A renewed certificate leads to chain validation errors for some clients.",
            "Connections fail before HTTP when the presented identity does not match the host.",
            "The client cannot verify the server's trust chain during encrypted setup.",
            "Calls fail at the handshake stage after a certificate swap.",
        ],
    },
    {
        "category": "QUEUE_BACKLOG",
        "check": "COMPARE_ARRIVAL_DRAIN_RATE",
        "reason": "Arrival versus drain rate and oldest-message age show whether work accumulates.",
        "primary": "Queue backlog: compare enqueue rate, consumer throughput, and oldest message age. Inspect consumer errors before adding workers, because failed messages can keep retrying.",
        "secondary": "Asynchronous processing delay: inspect pending job count and consumer lag over time. Separate a traffic surge from a stuck worker or poison message.",
        "tickets": [
            "Incoming tasks outnumber completed tasks and pending work keeps growing.",
            "The oldest queued job waits longer every hour despite a steady producer.",
            "Background messages pile up while workers finish fewer jobs than arrive.",
            "Async processing falls behind and the pending count climbs throughout the shift.",
            "Delivery lag increases because consumers cannot keep up with produced events.",
            "The work queue drains more slowly than new items are submitted.",
            "Jobs wait progressively longer as processing throughput drops below arrival rate.",
            "Backlogged events accumulate; the oldest item's age rises.",
        ],
    },
    {
        "category": "DISK_FULL",
        "check": "INSPECT_FILESYSTEM_USAGE",
        "reason": "Capacity and inode exhaustion require different remediation.",
        "primary": "Filesystem exhaustion: inspect free bytes and inodes on the exact mounted volume; identify growth source and retention settings before deleting files.",
        "secondary": "Storage write failure: correlate no-space errors with volume utilization and mount points. Check logs, temporary files, and inode usage.",
        "tickets": [
            "Writes fail with no space left on device while the application reads still work.",
            "The worker cannot persist checkpoints because its mounted volume is filled.",
            "New log segments fail to open when available filesystem capacity reaches zero.",
            "A batch job cannot save output, reporting storage exhausted on a mount.",
            "Only write operations fail; the data partition has no available space.",
            "The service logs ENOSPC and cannot create new files.",
            "A mounted disk has no room left for fresh artifacts or logs.",
            "Temporary file creation fails after the filesystem fills up.",
        ],
    },
    {
        "category": "DNS_FAILURE",
        "check": "INSPECT_NAME_RESOLUTION",
        "reason": "Resolver errors and DNS records distinguish name lookup from service failure.",
        "primary": "DNS resolution failure: query the name from the affected environment, inspect resolver configuration, TTL, and records. Do not assume the target service itself is unhealthy.",
        "secondary": "Hostname diagnosis: compare name lookup with direct IP reachability and check whether recent DNS changes propagated to each resolver.",
        "tickets": [
            "A service cannot resolve the host name for its database despite direct IP reachability.",
            "Requests fail before connecting because the configured domain returns no address.",
            "The client reports a name lookup error while the target process is healthy.",
            "Connections never start; hostname resolution intermittently returns NXDOMAIN.",
            "Some pods cannot map a service name to an IP after a zone update.",
            "The resolver cannot find the endpoint's address, though the machine answers by IP.",
            "An internal hostname no longer resolves for one workload.",
            "Traffic fails at address lookup before opening a socket.",
        ],
    },
    {
        "category": "RATE_LIMIT",
        "check": "INSPECT_QUOTA_HEADERS",
        "reason": "Quota responses and retry headers identify throttling versus timeouts.",
        "primary": "Rate limit response: inspect 429 counts, quota headers, and Retry-After guidance. Reduce request bursts or coordinate quota rather than retrying immediately.",
        "secondary": "Provider throttling: separate fast quota rejections from slow timeouts; inspect per-key limits and client retry behavior.",
        "tickets": [
            "The provider immediately rejects bursts with HTTP 429 responses.",
            "API requests exceed a per-key quota and receive a retry-after value.",
            "Calls fail fast due to throttling during traffic spikes.",
            "A remote endpoint refuses new requests when our request rate crosses its limit.",
            "Fast quota exceeded errors appear only during batch bursts.",
            "The client receives too many requests responses and a cooldown header.",
            "High request bursts trigger provider throttling instead of a network timeout.",
            "The service returns 429 and indicates when the client may try again.",
        ],
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "sample"
    passages: list[dict] = []
    partitions: dict[str, list[dict]] = {name: [] for name in ("train", "validation", "test")}
    qpartitions: dict[str, list[dict]] = {name: [] for name in ("validation", "test")}
    for family in FAMILIES:
        category = family["category"]
        primary, secondary = f"{category.lower()}-primary", f"{category.lower()}-secondary"
        passages.extend([
            {"id": primary, "category": category, "title": f"{category} incident guide",
             "text": family["primary"]},
            {"id": secondary, "category": category, "title": f"{category} diagnostic notes",
             "text": family["secondary"]},
        ])
        for index, ticket in enumerate(family["tickets"]):
            split = "train" if index < 5 else "validation" if index == 5 else "test"
            ticket_id = f"{category.lower()}-{index:02d}"
            partitions[split].append({
                "id": ticket_id, "group_id": ticket_id, "text": ticket,
                "category": category, "first_check": family["check"], "reason": family["reason"],
            })
            if split != "train":
                qpartitions[split].append({
                    "id": ticket_id, "text": ticket,
                    "relevance": {primary: 2, secondary: 1},
                })
    write_jsonl(root / "passages.jsonl", passages)
    for name, rows in partitions.items():
        write_jsonl(root / f"tickets.{name}.jsonl", rows)
    for name, rows in qpartitions.items():
        write_jsonl(root / f"queries.{name}.jsonl", rows)
    (root / "manifest.json").write_text(json.dumps({
        "origin": "Original fictional examples authored for ModelForge; no imported tickets or runbooks.",
        "license": "MIT (original project data)", "seed": 20260929,
        "split": "Five train, one validation, two test phrasing variants per family; groups are ticket IDs.",
        "counts": {"families": len(FAMILIES), "passages": len(passages),
                   **{name: len(rows) for name, rows in partitions.items()}},
        "limitations": "Small templated corpus; labels and guide prose share author and taxonomy."
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
