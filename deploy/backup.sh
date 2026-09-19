#!/usr/bin/env bash
# Backup (Phase 23).
#
#   deploy/backup.sh [destination-directory]
#
# Default destination: ./backups. Intended to run from cron on the host,
# after the close:
#
#   30 16 * * 1-5 cd /srv/indian-regime-trader && deploy/backup.sh >> /var/log/irt-backup.log 2>&1
#
# What is backed up, and why those and not others:
#
#   app-state  (the persisted system state)  -- UNRECOVERABLE if lost.
#       This is what the system believes it already did: the portfolio
#       snapshot and the last broker event it processed. Without it, a
#       restart cannot reconcile against the broker, and Phase 18's
#       entire recovery path has nothing to recover from. Backed up every
#       run.
#
#   model-registry (the approved model)      -- backed up every run.
#       Losing this does not lose money, but it loses the record of which
#       model was signed off and running -- and that record is what an
#       audit, or a post-incident reconstruction, actually needs. It is
#       not reconstructible: refitting produces a *different* model, not
#       this one, because the fit depends on the data as it stood and on
#       a seed nobody wrote down afterwards.
#
#   db-data    (Postgres)                    -- backed up every run.
#       See the note at the end about what actually lives here today.
#
#   app-logs   (the audit trail)             -- backed up every run.
#       Append-only in practice. Restored only to a *separate* location
#       for reading, never over a newer log: an audit trail you have
#       overwritten is no longer an audit trail.
#
#   app-data-cache                           -- NOT backed up, on purpose.
#       Reconstructible from the vendor. Backing it up would multiply the
#       size of every backup by the largest and least valuable thing on
#       the host, which is how backups quietly stop being taken.
#
# Restore is the other half and is documented in docs/OPERATIONS.md,
# including the instruction to actually test it -- an untested backup is
# a belief, not a backup.

set -euo pipefail

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-indian-regime-trader}"
DESTINATION="${1:-./backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TARGET="${DESTINATION}/${TIMESTAMP}"

# Read from the alpine image rather than from the host filesystem: volume
# paths under /var/lib/docker are an implementation detail of the storage
# driver, and reading them directly is how a backup script starts working
# on one host and silently producing empty archives on another.
HELPER_IMAGE="alpine:3.20"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

fail() {
    log "BACKUP FAILED: $*"
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "docker is not on PATH"

mkdir -p "${TARGET}"
log "backing up to ${TARGET}"

backup_volume() {
    local volume_suffix="$1"
    local archive_name="$2"
    local volume="${PROJECT_NAME}_${volume_suffix}"

    if ! docker volume inspect "${volume}" >/dev/null 2>&1; then
        fail "volume ${volume} does not exist (is the stack deployed, and is COMPOSE_PROJECT_NAME right?)"
    fi

    log "  ${volume} -> ${archive_name}"
    docker run --rm \
        --volume "${volume}:/source:ro" \
        --volume "$(cd "${TARGET}" && pwd):/backup" \
        "${HELPER_IMAGE}" \
        tar -czf "/backup/${archive_name}" -C /source . \
        || fail "could not archive ${volume}"
}

backup_volume "app-state" "app-state.tar.gz"
backup_volume "app-logs" "app-logs.tar.gz"
backup_volume "model-registry" "model-registry.tar.gz"

# Postgres gets a logical dump rather than a volume tarball. A tarball of
# a *running* database's data directory is a torn copy that may or may
# not restore; pg_dump produces a consistent snapshot by design.
#
# The credentials come from *inside* the container, not from this shell.
# `docker compose exec` does not forward the host's environment, so a
# host-side PGPASSWORD silently reaches nothing and pg_dump sits waiting
# for a password that a cron job will never type. Reading POSTGRES_USER /
# POSTGRES_PASSWORD / POSTGRES_DB from the db container's own environment
# (put there by deploy/db.env) also means this script needs no copy of the
# database credentials at all -- one fewer place for them to live.
log "  postgres -> postgres.sql.gz"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"
if [ -n "$(docker compose -f "${COMPOSE_FILE}" ps --status running --quiet db 2>/dev/null)" ]; then
    docker compose -f "${COMPOSE_FILE}" exec -T db sh -c \
        'PGPASSWORD="$POSTGRES_PASSWORD" pg_dump --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --clean --if-exists' \
        | gzip > "${TARGET}/postgres.sql.gz" \
        || fail "pg_dump failed"
else
    log "  postgres container is not running; skipping the dump"
    log "  (the manifest records which archives exist, so a restore does not assume one)"
fi

# --- Verify, then record ---------------------------------------------------
#
# A backup nobody verified is a backup nobody has. This is the cheap
# verification -- the archives exist, are non-empty, and are readable as
# gzip. The expensive one (restore into a scratch stack and start the
# service against it) is a documented monthly drill in docs/OPERATIONS.md,
# because only a restore proves a restore.
for archive in "${TARGET}"/*.gz; do
    [ -s "${archive}" ] || fail "${archive} is empty"
    gzip --test "${archive}" || fail "${archive} is not a readable gzip stream"
done

{
    echo "timestamp_utc=${TIMESTAMP}"
    echo "project=${PROJECT_NAME}"
    echo "host=$(hostname)"
    echo "files:"
    for archive in "${TARGET}"/*.gz; do
        echo "  $(basename "${archive}") $(wc -c < "${archive}") bytes"
    done
} > "${TARGET}/MANIFEST.txt"

# Backups contain positions, order history and the audit trail. They are
# as sensitive as the database itself and are readable by the owner only.
chmod -R go-rwx "${TARGET}"

log "backup complete: ${TARGET}"

# --- Retention -------------------------------------------------------------
# Keep 30 dated directories. Deliberately *not* unbounded: a disk that
# fills is an outage, and an outage caused by backups is an outage that
# also loses the next backup.
KEEP="${BACKUP_KEEP:-30}"
mapfile -t existing < <(find "${DESTINATION}" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#existing[@]}" -gt "${KEEP}" ]; then
    to_remove=$(( ${#existing[@]} - KEEP ))
    for (( i = 0; i < to_remove; i++ )); do
        log "pruning old backup ${existing[$i]}"
        rm -rf "${existing[$i]}"
    done
fi

# --- An honest note about what is in the database today --------------------
#
# storage/database.py is still a Phase 12 stub: the application does not
# yet read or write Postgres. The compose stack provisions it because the
# schema and the least-privilege role should exist and be exercised
# before anything depends on them, but today the dump above will be
# nearly empty and the state that actually matters is app-state.
#
# This is also why live.preflight's check 16 ("database backups") reads
# FAIL and should keep reading FAIL until storage/ is implemented,
# migrations exist, and a restore has been rehearsed. Do not mark that
# check passed because this script exists.
