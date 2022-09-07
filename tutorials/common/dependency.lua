includedirs { "../../contrib/Orochi/" }
files {"../../contrib/Orochi/Orochi/**.h", "../../contrib/Orochi/Orochi/**.cpp"}
files {"../../contrib/Orochi/contrib/**.h", "../../contrib/Orochi/contrib/**.cpp"}

hiprtroot = "../../"
if _OPTIONS["hiprtroot"] then
    hiprtroot = "../".._OPTIONS["hiprtroot"]
end

if os.istarget("windows") then
    links{ "version" }
    libdirs{hiprtroot.."/hiprt/win/"}
end
if os.ishost("linux") then
    links { "pthread", "dl" }
    libdirs{hiprtroot.."/hiprt/linux64/"}
end

files { hiprtroot.."/hiprt/*.h"}
links {"hiprt64"}
includedirs{ hiprtroot } 

