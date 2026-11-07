# GPU discipline for agents (and humans) sharing a machine

Timing is only meaningful if nothing else is on the GPU, and a shared machine is only safe if nobody touches what
they don't own. These rules come from incidents K-07, K-10, K-11 and K-12 in [starter-lessons.md](starter-lessons.md).

## Ownership

- Record in `CONSTRAINTS.md` which GPUs you own, by **physical index or PCI bus id**, and who owns the others
  (USER tag). Treat it as COACH precedence.
- Pin every GPU command:
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<owned index>`. Without the first variable, CUDA's default
  "fastest first" order can map index 0 to a different card than `nvidia-smi` shows.
- Never run on a GPU you don't own, including "just a quick test" or a test suite that might pick up a device.
  Stopping a service that holds your GPU is a plan change: raise it at C3 first, and stop rather than disable it.

## Locking and processes

- Hold an **exclusive lock** for every GPU command: `flock /tmp/<project>-gpu<N>.lock <command>`. Benchmarks,
  tests and profiles all take it.
- **One GPU process per agent**, none left in the background. Each torch+CUDA process reserves 10–25 GB of host
  virtual memory; under strict overcommit a few idle processes can exhaust the host commit limit and make
  unrelated spawns fail.
- Before launching, check headroom: GPU memory free, host `Committed_AS` vs `CommitLimit` (Linux
  `/proc/meminfo`), and that no other process of yours is alive.
- Kill your own debug processes when done; never kill another owner's process.

## Waiting and sequencing

- Wait on a PID (`wait <pid>`, a PID file), or on the lock itself (`flock` blocks until free).
- Don't wait with `pgrep -f <name>` when `<name>` appears in your own command line or another waiter's: it matches
  itself and never exits.
- Write a timeout and a log line into every wait so a stuck run is visible.

## Long runs and first requests

- Autotune on a short input first (same tuning keys), then run long inputs. A first long request can OOM in
  autotuning even when steady state fits.
- Record autotune picks per shape; pin them for production if processes can pick differently.

## Shared code and defaults

- Defaults in shared code match what production runs. New behaviour is opt-in (env var or flag) until confirmed and
  approved at C3.
- Review every subagent diff for default changes, removed flags and changed env-var meanings before merging.

## Reporting slips

Any slip (unlocked GPU use, unpinned device, a background process left alive) gets a ledger note with its time
window and a check of whether any timed run overlapped it, plus a candidate lesson. Harmless slips still count:
they're the cheap warning before an expensive one.
