import { defineConfig } from "eslint/config";
import next from "eslint-config-next";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig([{
    // Python virtualenvs, build output and sibling git worktrees are not our
    // source. Without this eslint walks into offline_backend/venv (torch ships
    // .mjs files), .next, and .claude/worktrees, turning 9 real findings in
    // app/lib/hooks into 42.
    ignores: [
        "**/venv/**",
        "**/.venv/**",
        "**/node_modules/**",
        "**/.next/**",
        ".claude/**",
        "**/__pycache__/**",
    ],
}, {
    extends: [...next],
}]);
