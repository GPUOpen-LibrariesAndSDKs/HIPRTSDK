project "16_fluid_simulation"
    cppdialect "C++17"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/**.h", "../common/**.cpp"} 
    files { "./**.h", "./**.cpp"} 

    includedirs{ "../../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"

    defines {"HIPRT_EXPORTS"}
