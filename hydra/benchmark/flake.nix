{
  description = "Hydra benchmark job wrapper";

  inputs.src.url = "path:../..";

  outputs =
    { src, ... }:
    {
      hydraJobs = src.hydraBenchmarkJobs;
    };
}
