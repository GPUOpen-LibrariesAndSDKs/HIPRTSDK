project "17_hiprt_hip"
	cppdialect "C++17"
    kind "ConsoleApp"
    location "../build"
  
    local hiproot = os.getenv("HIP_PATH")  
    if hiproot == nil or hiproot == '' then
        if os.ishost("linux") then
            hiproot = "/opt/rocm"
        end
    end

	if hiproot ~= nil then
		defines {"__USE_HIP__"}
		defines {"__HIP_PLATFORM_AMD__"}

		sysincludedirs {hiproot .. "/include/"}

		libdirs {hiproot .. "/lib/"}
		links { "amdhip64" }
		if os.istarget("windows") then
			libdirs {"../../hiprt/win/" }
		elseif os.ishost("linux") then
			libdirs { "../../hiprt/linux64/" }
		end
	end
	includedirs { "../../" }
	
    files { "./**.h", "./**.cpp"} 
	files { "../../hiprt/*.h"}

	links {"hiprt0200464"}
	targetdir "../dist/bin/%{cfg.buildcfg}"
