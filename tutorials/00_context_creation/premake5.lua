project "00_context_creation"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "./**.h", "./**.cpp"} 

    includedirs{ "../../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"

