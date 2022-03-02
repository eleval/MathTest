#include <DirectXMath/DirectXMath.h>

using namespace DirectX;

#include <Eigen/Eigen>
#include <glm/glm.hpp>


#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

std::vector<uint64_t> glmVec4DotTimes;
std::vector<uint64_t> eigenVec4DotTimes;
std::vector<uint64_t> dxVec4DotTimes;

std::vector<uint64_t> glmMat4MulTimes;
std::vector<uint64_t> eigenMat4MulTimes;
std::vector<uint64_t> dxMat4MulTimes;

std::vector<uint64_t> glmMat3MulTimes;
std::vector<uint64_t> eigenMat3MulTimes;
std::vector<uint64_t> dxMat3MulTimes;

std::vector<uint64_t> glmMat4InvTimes;
std::vector<uint64_t> eigenMat4InvTimes;
std::vector<uint64_t> dxMat4InvTimes;

std::vector<uint64_t> glmMat4Vec4MulTimes;
std::vector<uint64_t> eigenMat4Vec4MulTimes;
std::vector<uint64_t> dxMat4Vec4MulTimes;

std::vector<uint64_t> glmMat4Vec3MulTimes;
std::vector<uint64_t> eigenMat4Vec3MulTimes;
std::vector<uint64_t> dxMat4Vec3MulTimes;

std::vector<uint64_t> glmMat3Vec3MulTimes;
std::vector<uint64_t> eigenMat3Vec3MulTimes;
std::vector<uint64_t> dxMat3Vec3MulTimes;

constexpr const int SamplesCount = 1024;
constexpr const int RunsPerSample = 4096;

std::array<float, RunsPerSample> values;

constexpr const size_t biggerThanCachesize = 10 * 1024 * 1024;
long *p = new long[biggerThanCachesize];

void FlushCache()
{
	// When you want to "flush" cache. 
	for (int i = 0; i < biggerThanCachesize; i++)
	{
		p[i] = rand();
	}
}

void InitValues()
{
	for (int i = 0; i < RunsPerSample; ++i)
	{
		values[i] = static_cast<float>(rand());
	}
}

float RandF()
{
	return values[rand() % values.size()];
}

uint64_t GetTime()
{
	auto t0 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count();
}

void TestVec4Dot()
{
	glmVec4DotTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<float> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::vec4 a = { RandF(), RandF(), RandF(), RandF() };
			const glm::vec4 b = { RandF(), RandF(), RandF(), RandF() };
			ret[j] = glm::dot(a, b);
		}
		const uint64_t end = GetTime();
		glmVec4DotTimes[i] = end - start;
	}

	eigenVec4DotTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<float> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Vector4f a = { RandF(), RandF(), RandF(), RandF() };
			const Eigen::Vector4f b = { RandF(), RandF(), RandF(), RandF() };
			ret[j] = a.dot(b);
		}
		const uint64_t end = GetTime();
		eigenVec4DotTimes[i] = end - start;
	}

	dxVec4DotTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<float> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMVECTOR a = XMVectorSet(RandF(), RandF(), RandF(), RandF());
			const XMVECTOR b = XMVectorSet(RandF(), RandF(), RandF(), RandF());
			const XMVECTOR retV = XMVector4Dot(a, b);
			ret[j] = XMVectorGetX(retV);
		}
		const uint64_t end = GetTime();
		dxVec4DotTimes[i] = end - start;
	}

	std::sort(glmVec4DotTimes.begin(), glmVec4DotTimes.end());
	std::sort(eigenVec4DotTimes.begin(), eigenVec4DotTimes.end());
	std::sort(dxVec4DotTimes.begin(), dxVec4DotTimes.end());
}

void TestMat4Mul()
{
	glmMat4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::mat4x4> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat4x4 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const glm::mat4x4 b = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		glmMat4MulTimes[i] = end - start;
	}

	eigenMat4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Matrix4f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Matrix4f a{ { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			const Eigen::Matrix4f b{ { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		eigenMat4MulTimes[i] = end - start;
	}

	dxMat4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMMATRIX> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMMATRIX a = XMMatrixSet(RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF());
			const XMMATRIX b = XMMatrixSet(RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF());
			ret[j] = XMMatrixMultiply(a, b);
		}
		const uint64_t end = GetTime();
		dxMat4MulTimes[i] = end - start;
	}

	std::sort(glmMat4MulTimes.begin(), glmMat4MulTimes.end());
	std::sort(eigenMat4MulTimes.begin(), eigenMat4MulTimes.end());
	std::sort(dxMat4MulTimes.begin(), dxMat4MulTimes.end());
}

void TestMat3Mul()
{
	glmMat3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::mat3x3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat3x3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const glm::mat3x3 b = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		glmMat3MulTimes[i] = end - start;
	}

	eigenMat3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Matrix3f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Matrix3f a{ { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() } };
			const Eigen::Matrix3f b{ { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() } };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		eigenMat3MulTimes[i] = end - start;
	}

	dxMat3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMFLOAT3X3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMFLOAT3X3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const XMFLOAT3X3 b = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const XMMATRIX c = XMLoadFloat3x3(&a) * XMLoadFloat3x3(&b);
			XMStoreFloat3x3(&ret[j], c);
		}
		const uint64_t end = GetTime();
		dxMat3MulTimes[i] = end - start;
	}

	std::sort(glmMat3MulTimes.begin(), glmMat3MulTimes.end());
	std::sort(eigenMat3MulTimes.begin(), eigenMat3MulTimes.end());
	std::sort(dxMat3MulTimes.begin(), dxMat3MulTimes.end());
}

void TestMat4Inv()
{
	glmMat4InvTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::mat4x4> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat4x4 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			ret[i] = glm::inverse(a);
		}
		const uint64_t end = GetTime();
		glmMat4InvTimes[i] = end - start;
	}

	eigenMat4InvTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Matrix4f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Matrix4f a{ { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			ret[j] = a.inverse();
		}
		const uint64_t end = GetTime();
		eigenMat4InvTimes[i] = end - start;
	}

	dxMat4InvTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMMATRIX> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMMATRIX a = XMMatrixSet(RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF());
			ret[j] = XMMatrixInverse(nullptr, a);
		}
		const uint64_t end = GetTime();
		dxMat4InvTimes[i] = end - start;
	}

	std::sort(glmMat4InvTimes.begin(), glmMat4InvTimes.end());
	std::sort(eigenMat4InvTimes.begin(), eigenMat4InvTimes.end());
	std::sort(dxMat4InvTimes.begin(), dxMat4InvTimes.end());
}

void TestMat4Vec4Mul()
{
	glmMat4Vec4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::vec4> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat4x4 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const glm::vec4 b = { RandF(), RandF(), RandF(), RandF() };
			ret[i] = b * a;
		}
		const uint64_t end = GetTime();
		glmMat4Vec4MulTimes[i] = end - start;
	}

	eigenMat4Vec4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Vector4f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Matrix4f a{ { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			const Eigen::Vector4f b = { RandF(), RandF(), RandF(), RandF() };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		eigenMat4Vec4MulTimes[i] = end - start;
	}

	dxMat4Vec4MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMVECTOR> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMMATRIX a = XMMatrixSet(RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF());
			const XMVECTOR b = XMVectorSet(RandF(), RandF(), RandF(), RandF());
			ret[j] = XMVector4Transform(b, a);
		}
		const uint64_t end = GetTime();
		dxMat4Vec4MulTimes[i] = end - start;
	}

	std::sort(glmMat4Vec4MulTimes.begin(), glmMat4Vec4MulTimes.end());
	std::sort(eigenMat4Vec4MulTimes.begin(), eigenMat4Vec4MulTimes.end());
	std::sort(dxMat4Vec4MulTimes.begin(), dxMat4Vec4MulTimes.end());
}

void TestMat4Vec3Mul()
{
	glmMat4Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::vec3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat4x3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const glm::vec3 b = { RandF(), RandF(), RandF() };
			ret[i] = b * a;
		}
		const uint64_t end = GetTime();
		glmMat4Vec3MulTimes[i] = end - start;
	}

	eigenMat4Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Vector3f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			Eigen::MatrixXf a;
			a.resize(4, 3);
			a = { { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			//const Eigen::Matrix34f a{ { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF(), RandF() } };
			const Eigen::Vector3f b = { RandF(), RandF(), RandF() };
			//ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		eigenMat4Vec3MulTimes[i] = end - start;
	}

	dxMat4Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMFLOAT3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMFLOAT4X3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const XMFLOAT3 b = { RandF(), RandF(), RandF() };
			const XMMATRIX a2 = XMLoadFloat4x3(&a);
			const XMVECTOR b2 = XMLoadFloat3(&b);
			const XMVECTOR c = XMVector3Transform(b2, a2);
			XMStoreFloat3(&ret[j], c);
		}
		const uint64_t end = GetTime();
		dxMat4Vec3MulTimes[i] = end - start;
	}

	std::sort(glmMat4Vec3MulTimes.begin(), glmMat4Vec3MulTimes.end());
	std::sort(eigenMat4Vec3MulTimes.begin(), eigenMat4Vec3MulTimes.end());
	std::sort(dxMat4Vec3MulTimes.begin(), dxMat4Vec3MulTimes.end());
}

void TestMat3Vec3Mul()
{
	glmMat3Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<glm::vec3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const glm::mat3x3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const glm::vec3 b = { RandF(), RandF(), RandF() };
			ret[i] = b * a;
		}
		const uint64_t end = GetTime();
		glmMat3Vec3MulTimes[i] = end - start;
	}

	eigenMat3Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<Eigen::Vector3f> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const Eigen::Matrix3f a{ { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() }, { RandF(), RandF(), RandF() } };
			const Eigen::Vector3f b = { RandF(), RandF(), RandF() };
			ret[j] = a * b;
		}
		const uint64_t end = GetTime();
		eigenMat3Vec3MulTimes[i] = end - start;
	}

	dxMat3Vec3MulTimes.resize(SamplesCount);
	FlushCache();
	for (int i = 0; i < SamplesCount; ++i)
	{
		std::vector<XMFLOAT3> ret(RunsPerSample);
		const uint64_t start = GetTime();
		for (int j = 0; j < RunsPerSample; ++j)
		{
			const XMFLOAT3X3 a = { RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF(), RandF() };
			const XMFLOAT3 b = { RandF(), RandF(), RandF() };
			const XMMATRIX a2 = XMLoadFloat3x3(&a);
			const XMVECTOR b2 = XMLoadFloat3(&b);
			const XMVECTOR c = XMVector3Transform(b2, a2);
			XMStoreFloat3(&ret[j], c);
		}
		const uint64_t end = GetTime();
		dxMat3Vec3MulTimes[i] = end - start;
	}

	std::sort(glmMat3Vec3MulTimes.begin(), glmMat3Vec3MulTimes.end());
	std::sort(eigenMat3Vec3MulTimes.begin(), eigenMat3Vec3MulTimes.end());
	std::sort(dxMat3Vec3MulTimes.begin(), dxMat3Vec3MulTimes.end());
}

int main(int argc, const char* argv[])
{
	InitValues();
	TestVec4Dot();
	TestMat4Mul();
	TestMat3Mul();
	TestMat4Inv();
	TestMat4Vec4Mul();
	TestMat4Vec3Mul();
	TestMat3Vec3Mul();

	std::cout << "Operation" << "\t\t" << "Min" << "\t" << "Max" << "\t" << "Med" << std::endl;
	
	std::cout << "glm Vec4Dot" << "\t\t" << glmVec4DotTimes.front() << "\t" << glmVec4DotTimes.back() << "\t" << glmVec4DotTimes[glmVec4DotTimes.size() / 2] << std::endl;
	std::cout << "Eigen Vec4Dot" << "\t\t" << eigenVec4DotTimes.front() << "\t" << eigenVec4DotTimes.back() << "\t" << eigenVec4DotTimes[glmVec4DotTimes.size() / 2] << std::endl;
	std::cout << "DX Vec4Dot" << "\t\t" << dxVec4DotTimes.front() << "\t" << dxVec4DotTimes.back() << "\t" << dxVec4DotTimes[glmVec4DotTimes.size() / 2] << std::endl;

	std::cout << "glm Mat4Mul" << "\t\t" << glmMat4MulTimes.front() << "\t" << glmMat4MulTimes.back() << "\t" << glmMat4MulTimes[glmMat4MulTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat4Mul" << "\t\t" << eigenMat4MulTimes.front() << "\t" << eigenMat4MulTimes.back() << "\t" << eigenMat4MulTimes[eigenMat4MulTimes.size() / 2] << std::endl;
	std::cout << "DX Mat4Mul" << "\t\t" << dxMat4MulTimes.front() << "\t" << dxMat4MulTimes.back() << "\t" << dxMat4MulTimes[dxMat4MulTimes.size() / 2] << std::endl;

	std::cout << "glm Mat3Mul" << "\t\t" << glmMat3MulTimes.front() << "\t" << glmMat3MulTimes.back() << "\t" << glmMat3MulTimes[glmMat3MulTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat3Mul" << "\t\t" << eigenMat3MulTimes.front() << "\t" << eigenMat3MulTimes.back() << "\t" << eigenMat3MulTimes[eigenMat3MulTimes.size() / 2] << std::endl;
	std::cout << "DX Mat3Mul" << "\t\t" << dxMat3MulTimes.front() << "\t" << dxMat3MulTimes.back() << "\t" << dxMat3MulTimes[dxMat3MulTimes.size() / 2] << std::endl;

	std::cout << "glm Mat4Inv" << "\t\t" << glmMat4InvTimes.front() << "\t" << glmMat4InvTimes.back() << "\t" << glmMat4InvTimes[glmMat4InvTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat4Inv" << "\t\t" << eigenMat4InvTimes.front() << "\t" << eigenMat4InvTimes.back() << "\t" << eigenMat4InvTimes[eigenMat4InvTimes.size() / 2] << std::endl;
	std::cout << "DX Mat4Inv" << "\t\t" << dxMat4InvTimes.front() << "\t" << dxMat4InvTimes.back() << "\t" << dxMat4InvTimes[dxMat4InvTimes.size() / 2] << std::endl;

	std::cout << "glm Mat4Vec4Mul" << "\t\t" << glmMat4Vec4MulTimes.front() << "\t" << glmMat4Vec4MulTimes.back() << "\t" << glmMat4Vec4MulTimes[glmMat4Vec4MulTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat4Vec4Mul" << "\t" << eigenMat4Vec3MulTimes.front() << "\t" << eigenMat4Vec3MulTimes.back() << "\t" << eigenMat4Vec3MulTimes[eigenMat4Vec3MulTimes.size() / 2] << std::endl;
	std::cout << "DX Mat4Vec4Mul" << "\t\t" << dxMat4Vec3MulTimes.front() << "\t" << dxMat4Vec3MulTimes.back() << "\t" << dxMat4Vec3MulTimes[dxMat4Vec3MulTimes.size() / 2] << std::endl;

	std::cout << "glm Mat4Vec3Mul" << "\t\t" << glmMat4Vec3MulTimes.front() << "\t" << glmMat4Vec3MulTimes.back() << "\t" << glmMat4Vec3MulTimes[glmMat4Vec3MulTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat4Vec3Mul" << "\t" << eigenMat4Vec3MulTimes.front() << "\t" << eigenMat4Vec3MulTimes.back() << "\t" << eigenMat4Vec3MulTimes[eigenMat4Vec3MulTimes.size() / 2] << std::endl;
	std::cout << "DX Mat4Vec3Mul" << "\t\t" << dxMat4Vec3MulTimes.front() << "\t" << dxMat4Vec3MulTimes.back() << "\t" << dxMat4Vec3MulTimes[dxMat4Vec3MulTimes.size() / 2] << std::endl;

	std::cout << "glm Mat3Vec3Mul" << "\t\t" << glmMat3Vec3MulTimes.front() << "\t" << glmMat3Vec3MulTimes.back() << "\t" << glmMat3Vec3MulTimes[glmMat3Vec3MulTimes.size() / 2] << std::endl;
	std::cout << "Eigen Mat3Vec3Mul" << "\t" << eigenMat3Vec3MulTimes.front() << "\t" << eigenMat3Vec3MulTimes.back() << "\t" << eigenMat3Vec3MulTimes[eigenMat3Vec3MulTimes.size() / 2] << std::endl;
	std::cout << "DX Mat3Vec3Mul" << "\t\t" << dxMat3Vec3MulTimes.front() << "\t" << dxMat3Vec3MulTimes.back() << "\t" << dxMat3Vec3MulTimes[dxMat3Vec3MulTimes.size() / 2] << std::endl;

	delete[] p;

	system("pause");
	return 0;
}