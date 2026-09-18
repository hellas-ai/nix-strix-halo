{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.boot.multikernel;
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types
    ;

  instanceType = types.submodule (
    { name, ... }:
    {
      options = {
        enable = mkEnableOption "the ${name} multikernel instance" // {
          default = true;
        };

        id = mkOption {
          type = types.nullOr (types.ints.between 1 511);
          default = null;
          description = "Stable multikernel instance ID; null lets Kerf allocate one.";
        };

        cpus = mkOption {
          type = types.str;
          example = "12-13,28-29";
          description = "Physical APIC IDs assigned to this instance.";
        };

        memory = mkOption {
          type = types.str;
          example = "2GB";
          description = "Contiguous memory allocated to this instance from the pool.";
        };

        devices = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "Kerf device aliases exclusively assigned to this instance.";
        };

        kernel = mkOption {
          type = types.str;
          default = cfg.spawnKernel;
          defaultText = lib.literalExpression "config.boot.multikernel.spawnKernel";
          description = "ELF vmlinux or x86 bzImage loaded into the instance.";
        };

        initrd = mkOption {
          type = types.nullOr types.str;
          default = cfg.spawnInitrd;
          defaultText = lib.literalExpression "config.boot.multikernel.spawnInitrd";
          description = "Optional initramfs passed to the spawn kernel.";
        };

        entrypoint = mkOption {
          type = types.str;
          default = "/bin/multikernel-demo";
          description = "Spawn initramfs entrypoint, interpreted by kerf-init.";
        };

        kernelParams = mkOption {
          type = types.listOf types.str;
          default = [
            "init=/init"
            "console=mktty0"
            "loglevel=6"
            "panic=-1"
            "pci=nobar"
            "random.trust_cpu=on"
            "sysrq_always_enabled=1"
          ];
          description = "Kernel command-line parameters for the spawn kernel.";
        };

        prepareAtBoot = mkOption {
          type = types.bool;
          default = false;
          description = "Create the instance and load its kernel during host boot.";
        };

        startAtBoot = mkOption {
          type = types.bool;
          default = false;
          description = "Boot the prepared spawn kernel during host boot.";
        };
      };
    }
  );

  enabledInstances = lib.filterAttrs (_: instance: instance.enable) cfg.instances;
  instanceNames = builtins.attrNames enabledInstances;
  kerf = lib.getExe cfg.package;

  poolArgs = [
    kerf
    "init"
    "--cpus=${cfg.pool.cpus}"
    "--memory=${cfg.pool.memory}"
  ]
  ++ lib.optional (cfg.pool.devices != [ ]) "--devices=${lib.concatStringsSep "," cfg.pool.devices}";

  instanceCmdline =
    name: instance:
    lib.concatStringsSep " " (
      instance.kernelParams
      ++ [
        "multikernel.demo.name=${name}"
        ''kerf.entrypoint="${instance.entrypoint}"''
      ]
    );

  createArgs =
    name: instance:
    [
      kerf
      "create"
      name
      "--cpus=${instance.cpus}"
      "--memory=${instance.memory}"
    ]
    ++ lib.optional (instance.id != null) "--id=${toString instance.id}"
    ++ lib.optional (instance.devices != [ ]) "--devices=${lib.concatStringsSep "," instance.devices}";

  loadArgs =
    name: instance:
    [
      kerf
      "load"
      name
      "--kernel=${instance.kernel}"
      "--cmdline=${instanceCmdline name instance}"
    ]
    ++ lib.optional (instance.initrd != null) "--initrd=${instance.initrd}";

  mkInstanceService = name: instance: {
    description = "Prepare multikernel instance ${name}";
    after = [ "multikernel-pool.service" ];
    requires = [ "multikernel-pool.service" ];
    wantedBy = lib.optional (instance.prepareAtBoot || instance.startAtBoot) "multi-user.target";
    path = [ pkgs.coreutils ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      set -eu
      instance_dir=/sys/fs/multikernel/instances/${name}

      if [ ! -d "$instance_dir" ]; then
        ${lib.escapeShellArgs (createArgs name instance)}
      fi

      status=$(cat "$instance_dir/status")
      case "$status" in
        ready)
          ${lib.escapeShellArgs (loadArgs name instance)}
          ;;
        loaded|active)
          ;;
        *)
          echo "multikernel instance ${name} has unexpected status: $status" >&2
          exit 1
          ;;
      esac

      ${lib.optionalString instance.startAtBoot ''
        status=$(cat "$instance_dir/status")
        if [ "$status" = loaded ]; then
          ${lib.escapeShellArgs [
            kerf
            "exec"
            name
          ]}
        fi
      ''}
    '';
  };

  instanceServices = lib.mapAttrs' (
    name: instance: lib.nameValuePair "multikernel-instance-${name}" (mkInstanceService name instance)
  ) enabledInstances;

  knownNames =
    if instanceNames == [ ] then
      "__no_declarative_instances__"
    else
      lib.concatStringsSep "|" instanceNames;
  multikernelctl = pkgs.writeShellApplication {
    name = "multikernelctl";
    runtimeInputs = [
      cfg.package
      pkgs.coreutils
      pkgs.systemd
    ];
    text = ''
      set -euo pipefail

      command="''${1:-show}"
      instance="''${2:-}"

      require_instance() {
        if [ -z "$instance" ]; then
          echo "usage: multikernelctl $command <instance>" >&2
          exit 2
        fi
        case "$instance" in
          ${knownNames}) ;;
          *)
            echo "unknown declarative instance: $instance" >&2
            echo "known instances: ${lib.concatStringsSep " " instanceNames}" >&2
            exit 2
            ;;
        esac
      }

      case "$command" in
        prepare)
          require_instance
          systemctl start "multikernel-instance-$instance.service"
          ;;
        start)
          require_instance
          systemctl start "multikernel-instance-$instance.service"
          exec kerf exec "$instance"
          ;;
        console)
          require_instance
          exec kerf console "$instance"
          ;;
        stop)
          require_instance
          exec kerf kill "$instance"
          ;;
        force-stop)
          require_instance
          exec kerf kill --force "$instance"
          ;;
        unload)
          require_instance
          exec kerf unload "$instance"
          ;;
        delete)
          require_instance
          kerf delete "$instance"
          systemctl stop "multikernel-instance-$instance.service"
          ;;
        show)
          if [ -n "$instance" ]; then
            exec kerf show "$instance"
          else
            exec kerf show
          fi
          ;;
        reset-pool)
          kerf init --cpus=none --memory=none
          systemctl stop multikernel-pool.service
          ;;
        *)
          echo "usage: multikernelctl {prepare|start|console|stop|force-stop|unload|delete|show|reset-pool} [instance]" >&2
          exit 2
          ;;
      esac
    '';
  };
in
{
  options.boot.multikernel = {
    enable = mkEnableOption "bare-metal multikernel Linux";

    selectHostKernel = mkOption {
      type = types.bool;
      default = true;
      description = "Select the packaged multikernel kernel as boot.kernelPackages.";
    };

    hostKernelPackages = mkOption {
      type = types.raw;
      default = pkgs.linuxPackages_multikernel;
      defaultText = lib.literalExpression "pkgs.linuxPackages_multikernel";
      description = "Linux package set used by the host kernel.";
    };

    package = mkOption {
      type = types.package;
      default = pkgs.kerf-multikernel;
      defaultText = lib.literalExpression "pkgs.kerf-multikernel";
      description = "Kerf multikernel lifecycle manager.";
    };

    spawnKernel = mkOption {
      type = types.str;
      default = "${cfg.hostKernelPackages.kernel.dev}/vmlinux";
      defaultText = lib.literalExpression ''"${config.boot.multikernel.hostKernelPackages.kernel.dev}/vmlinux"'';
      description = "Default kernel image loaded into spawn instances.";
    };

    spawnInitrd = mkOption {
      type = types.nullOr types.str;
      default = "${pkgs.multikernel-demo-initrd}/initrd";
      defaultText = lib.literalExpression ''"${pkgs.multikernel-demo-initrd}/initrd"'';
      description = "Default self-contained spawn initramfs.";
    };

    pool = {
      cpus = mkOption {
        type = types.str;
        example = "12-15,28-31";
        description = "Physical APIC IDs moved from the host into the multikernel pool.";
      };

      memory = mkOption {
        type = types.str;
        example = "8GB";
        description = "Runtime contiguous-memory pool managed by the host kernel.";
      };

      devices = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Kerf device aliases moved into the multikernel pool.";
      };

      prepareAtBoot = mkOption {
        type = types.bool;
        default = false;
        description = "Create the resource pool during host boot.";
      };
    };

    instances = mkOption {
      type = types.attrsOf instanceType;
      default = { };
      description = "Declarative spawn-kernel instances.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = builtins.match "[0-9,-]+" cfg.pool.cpus != null;
        message = "boot.multikernel.pool.cpus must be a Kerf APIC-ID list such as 12-15,28-31";
      }
      {
        assertion =
          builtins.match "[0-9]+(KB|MB|GB|TB|KiB|MiB|GiB|TiB|B)?(@[0-9]+)?(,[0-9]+(KB|MB|GB|TB|KiB|MiB|GiB|TiB|B)?(@[0-9]+)?)*" cfg.pool.memory
          != null;
        message = "boot.multikernel.pool.memory must be a Kerf size such as 8GB";
      }
    ]
    ++ lib.concatMap (
      name:
      let
        instance = enabledInstances.${name};
      in
      [
        {
          assertion = builtins.match "[A-Za-z0-9_.-]+" name != null;
          message = "multikernel instance names may contain only letters, digits, dot, underscore and dash";
        }
        {
          assertion = builtins.match "[0-9,-]+" instance.cpus != null;
          message = "boot.multikernel.instances.${name}.cpus must be an APIC-ID list";
        }
      ]
    ) instanceNames;

    boot.kernelPackages = mkIf cfg.selectHostKernel (lib.mkOverride 50 cfg.hostKernelPackages);
    boot.kernel.sysctl."kernel.kexec_load_disabled" = 0;

    environment.systemPackages = [
      cfg.package
      multikernelctl
      pkgs.dtc
    ];

    systemd.services = {
      multikernel-pool = {
        description = "Prepare the bare-metal multikernel resource pool";
        wantedBy = lib.optional cfg.pool.prepareAtBoot "multi-user.target";
        before = map (name: "multikernel-instance-${name}.service") instanceNames;
        path = [ pkgs.coreutils ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
        };
        script = ''
          set -eu
          ${lib.escapeShellArgs poolArgs}
          test -d /sys/fs/multikernel/instances
        '';
      };
    }
    // instanceServices;
  };
}
