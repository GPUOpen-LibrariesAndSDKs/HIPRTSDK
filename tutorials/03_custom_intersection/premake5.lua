project "03_custom_intersection"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/**.h", "../common/**.cpp"} 
    files { "./**.h", "./**.cpp"} 
	files {"../../contrib/Orochi/Orochi/**.h", "../../contrib/Orochi/Orochi/**.cpp"}
	files {"../../contrib/Orochi/contrib/**.h", "../../contrib/Orochi/contrib/**.cpp"}
    
	includedirs{ "../../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"