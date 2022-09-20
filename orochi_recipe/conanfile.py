from conans import ConanFile, CMake, tools

required_conan_version = ">=1.43.0"


class OrochiConan(ConanFile):
    name = "orochi"
    homepage = "https://github.com/GPUOpen-LibrariesAndSDKs/Orochi"
    url = "https://github.com/GPUOpen-LibrariesAndSDKs/Orochi.git"

    description = "Orochi is a library loading HIP and CUDA APIs dynamically, allowing the user to switch APIs at runtime."
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"

    exports_sources = "CMakeLists.txt"

    generators = "cmake", "cmake_find_package"

    version = "1.00"

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    @property
    def _build_subfolder(self):
        return "build_subfolder"

    def source(self):

        if self.version == "latest":
            git = tools.Git(folder=self._source_subfolder)
            git.clone(self.url, "main")
        else:
            tools.get(
                **self.conan_data["sources"][self.version],
                destination=self._source_subfolder,
                strip_root=True
            )

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_folder=self._build_subfolder)
        cmake.build()

    def package(self):
        self.copy(pattern="LICENSE*", src=self._source_subfolder, dst="licenses")

        cmake = CMake(self)
        cmake.configure(build_folder=self._build_subfolder)
        cmake.install()

        self.copy(
            pattern="*.dll",
            src=self._source_subfolder + "/contrib/bin/win64",
            dst="bin",
        )

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "orochi"
        self.cpp_info.names["cmake_find_package_multi"] = "orochi"
        self.cpp_info.names["pkg_config"] = "orochi"

        self.cpp_info.libs = ["orochi"]

        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["dl", "m", "pthread"])

        if self.settings.os == "Windows":
            self.cpp_info.system_libs.extend(["version"])
