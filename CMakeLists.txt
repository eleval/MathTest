cmake_minimum_required(VERSION 3.21.1)

project(MathTest)

set(CMAKE_PREFIX_PATH "G:/Libraries/install")

find_package(glm REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(DirectXMath REQUIRED)

set(SRC
	main.cpp
)

add_executable(MathTest ${SRC})

target_link_libraries(MathTest
	PUBLIC glm::glm
	PUBLIC Eigen3::Eigen
	PUBLIC Microsoft::DirectXMath
)