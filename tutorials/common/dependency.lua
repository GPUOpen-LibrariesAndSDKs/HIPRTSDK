includedirs { "../../contrib/Orochi/" }
files {"../../contrib/Orochi/Orochi/**.h", "../../contrib/Orochi/Orochi/**.cpp"}
files {"../../contrib/Orochi/contrib/**.h", "../../contrib/Orochi/contrib/**.cpp"}

if os.istarget("windows") then
    links{ "version" }
    libdirs{"../../hiprt/win/"}
end
if os.ishost("linux") then
    links { "pthread", "dl" }
    libdirs{"../../hiprt/linux64/"}
end

files { "../../hiprt/*.h"}
links {"hiprt0200464"}

