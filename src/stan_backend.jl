import Stan

const STAN_MODEL_FILE = normpath(joinpath(@__DIR__, "stan", "prophet.stan"))
const REQUIRED_CMDSTAN_VERSION = v"2.37.0"

function stan_model_file()
    isfile(STAN_MODEL_FILE) || error("Stan model file not found at $STAN_MODEL_FILE")
    return STAN_MODEL_FILE
end

function stan_backend_module()
    return Stan
end

function cmdstan_home()
    home = get(ENV, "CMDSTAN", get(ENV, "JULIA_CMDSTAN_HOME", ""))
    isempty(home) && error("CMDSTAN or JULIA_CMDSTAN_HOME must point to CmdStan.")
    isdir(home) || error("CmdStan directory does not exist: $home")
    return home
end

function cmdstan_stanc()
    exe = Sys.iswindows() ? "stanc.exe" : "stanc"
    path = joinpath(cmdstan_home(), "bin", exe)
    isfile(path) || error("CmdStan stanc executable not found at $path")
    return path
end

function cmdstan_version()
    output = readchomp(`$(cmdstan_stanc()) --version`)
    m = match(r"([0-9]+)\.([0-9]+)\.([0-9]+)", output)
    m === nothing && error("Unable to parse CmdStan version from: $output")
    return VersionNumber(parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3]))
end
