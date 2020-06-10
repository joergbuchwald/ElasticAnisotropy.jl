using ElasticAnisotropy
using Test

O = OrthoModel(8e9,4e9, 0.35, 0.5, 1.48148e9)
n = [-0.573576436, 0.0, 0.819152044]
R = getRmatrix(n)
C_voigt = generateVoigtMatrix(O)
c_tensor = VoigtMatrix2Tensor(C_voigt)
c_tensor_R = rotateTensor(c_tensor,n)
c_tensor_R2 = rotateTensor(c_tensor,R)
C_voigt_R = Tensor2VoigtMatrix(c_tensor_R)
C_voigt_R2 = Tensor2VoigtMatrix(c_tensor_R2)


@testset "ElasticAnisotropy.jl" begin
    @test C_voigt_R == C_voigt_R2
end
