{
  gnugrep,
  patch,
  pythonPackages,
  sglang-glm53-rocm,
  observerName,
  observerDirectory,
  observerPatchName,
  observerTestName,
}:

# Explicit experimental packages use the complete base runtime gates first.
# The final Python-only observer patch then gets its own installed-source gate;
# the base capture plan remains an exact account of the pre-observer sources.
let
  sitePackages = pythonPackages.python.sitePackages;
in
sglang-glm53-rocm.overrideAttrs (old: {
  pname = "${old.pname}-${observerName}";
  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ patch gnugrep ];
  postFixup = old.postFixup + ''
    observer_log="$TMPDIR/glm53-${observerName}-patch.log"
    ${patch}/bin/patch --batch --fuzz=0 -p1 -d "$out/${sitePackages}" \
      < ${observerDirectory}/${observerPatchName} > "$observer_log"
    cat "$observer_log"
    if ${gnugrep}/bin/grep -Eiq '\b(offset|fuzz)\b' "$observer_log"; then
      echo "GLM observer patch must apply with zero fuzz and zero offset" >&2
      exit 1
    fi
    ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${observerDirectory}/${observerTestName} \
      --source "$out/${sitePackages}/sglang" --installed
  '';
  passthru = (old.passthru or { }) // {
    glm53Observer = {
      inherit observerName observerDirectory;
      baselineRuntime = sglang-glm53-rocm;
      defaultEnabled = false;
      numericalAndModelEquivalence = "UNASSESSED";
    };
  };
})
