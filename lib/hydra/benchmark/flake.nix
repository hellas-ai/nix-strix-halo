{
  description = "Hydra benchmark job selector";

  inputs.src.url = "path:../../..";

  outputs =
    { src, ... }:
    {
      hydraJobs = src.hydraBenchmarkJobs;
    };
}
