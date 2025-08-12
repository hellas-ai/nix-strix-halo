# Strixtop - System monitoring dashboard for Strix
{
  lib,
  writeShellScriptBin,
  tmux,
  amdgpu_top,
  btop,
  nvtopPackages,
  ec-su-axb35-monitor,
  systemd,
}: let
  tmuxExe = lib.getExe tmux;
  amdgpuTopExe = lib.getExe amdgpu_top;
  btopExe = lib.getExe btop;
  nvtopExe = lib.getExe nvtopPackages.amd;
  ecMonitorExe = lib.getExe ec-su-axb35-monitor;
  journalctlExe = "${systemd}/bin/journalctl";
in
  writeShellScriptBin "strixtop" ''
    # strixtop - System monitoring dashboard for Strix
    # Creates a tmux session with multiple monitoring tools

    SESSION_NAME="strixtop"

    # Check if session exists
    ${tmuxExe} has-session -t "$SESSION_NAME" 2>/dev/null

    if [ $? != 0 ]; then
        # Create new session with amdgpu_top on the left
        ${tmuxExe} new-session -d -s "$SESSION_NAME" -n "monitoring" '${amdgpuTopExe}'

        # Split vertically for right side
        ${tmuxExe} split-window -h -t "$SESSION_NAME:monitoring"

        # Right pane: btop at the top
        ${tmuxExe} send-keys -t "$SESSION_NAME:monitoring.1" '${btopExe}' C-m

        # Split right pane horizontally for middle section
        ${tmuxExe} split-window -v -t "$SESSION_NAME:monitoring.1" -p 70

        # In the new pane (monitoring.2), run ec_su_axb35_monitor
        ${tmuxExe} send-keys -t "$SESSION_NAME:monitoring.2" '${ecMonitorExe}' C-m

        # Split this pane vertically for nvtop
        ${tmuxExe} split-window -h -t "$SESSION_NAME:monitoring.2"
        ${tmuxExe} send-keys -t "$SESSION_NAME:monitoring.3" '${nvtopExe}' C-m

        # Split the bottom for journalctl (10 lines high)
        ${tmuxExe} split-window -v -t "$SESSION_NAME:monitoring.2" -l 10
        ${tmuxExe} send-keys -t "$SESSION_NAME:monitoring.4" '${journalctlExe} -xef' C-m

        # Set focus to the main pane
        ${tmuxExe} select-pane -t "$SESSION_NAME:monitoring.0"
    fi

    # Attach to session
    ${tmuxExe} attach-session -t "$SESSION_NAME"
  ''
