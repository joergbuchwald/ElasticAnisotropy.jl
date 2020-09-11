module ElasticAnisotropy

export getRmatrix, OrthoModel, TransIsomodel, IsoModel, rotateTensor, Tensor2VoigtMatrix, VoigtMatrix2Tensor, generateVoigtMatrix

using LinearAlgebra

struct OrthoModel
    moduli::NamedTuple{(:E_1, :E_2, :E_3, :ν_12, :ν_23, :ν_13, :G_12, :G_23, :G_13), Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}
end

struct TransIsoModel
    moduli::NamedTuple{(:E_i,:E_a,:ν_i,:ν_ia,:G_a), Tuple{Float64,Float64,Float64,Float64,Float64}}
end

struct IsoModel
    moduli::NamedTuple{(:E,:ν), Tuple{Float64,Float64}}
end

function generateVoigtMatrix(p::OrthoModel)
    E_1, E_2, E_3, ν_12, ν_23, ν_13, G_12, G_23, G_13 = p.moduli
    ν_21 = ν_12 * E_2 / E_1
    ν_32 = ν_23 * E_3 / E_2
    ν_31 = ν_13 * E_3 / E_1
    D = (1 - ν_12*ν_21-ν_23*ν_32-ν_31*ν_13-2*ν_12*ν_23*ν_31)/(E_1 * E_2 * E_3)
    C = [(1-ν_23*ν_32)/(E_2*E_3*D) (ν_21+ν_31*ν_23)/(E_2*E_3*D) (ν_31+ν_21*ν_32)/(E_2*E_3*D) 0 0 0; (ν_12+ν_13*ν_32)/(E_1*E_3*D) (1-ν_31*ν_13)/(E_1*E_3*D) (ν_32+ν_31*ν_12)/(E_1*E_3*D) 0 0 0; (ν_13+ν_12*ν_23)/(E_1*E_2*D) (ν_23+ν_13*ν_21)/(E_1*E_2*D) (1-ν_12*ν_21)/(E_1*E_2*D) 0 0 0; 0 0 0 G_23 0 0; 0 0 0 0 G_31 0; 0 0 0 0 0 G_12]
    return C
end

function generateVoigtMatrix(p::TransIsoModel)
    E_i, E_a, ν_i, ν_ia, G_a = p.moduli
    G_i = E_i / (2.0*(1. + ν_i))
    ν_ai = ν_ia * E_a / E_i
    D = ((1.0+ν_i)*(1.0-ν_i-2*ν_ia * ν_ai))/(E_i^2*E_a)
    a_ii = (1-ν_ia*ν_ai)/(E_i*E_a*D)
    a_ai = (1-ν_i^2)/(E_i^2*D)
    b_ii = (ν_i+ν_ia*ν_ai)/(E_i*E_a*D)
    b_ai = (ν_ia*(1+ν_i))/(E_i^2*D)
    c_ii = E_i/(2*(1+ν_i))
    c_ai = G_a
    C = [a_ii b_ii b_ai 0 0 0; b_ii a_ii b_ai 0 0 0; b_ai b_ai a_ai 0 0 0; 0 0 0 c_ai 0 0; 0 0 0 0 c_ai 0; 0 0 0 0 0 c_ii]
    return C
end

function generateVoigtMatrix(p::IsoModel)
    E, ν = p.moduli
    prefac = E/((1+ν)*(1-2*ν))
    a_ii = 1-ν
    a_ij = ν
    b_ii = (1-2*ν)/2
    C = prefac .* [a_ii a_ij a_ij 0 0 0; a_ij a_ii a_ij 0 0 0; a_ij a_ij a_ii 0 0 0; 0 0 0 b_ii 0 0; 0 0 0 0 b_ii 0; 0 0 0 0 0 b_ii]
    return C
end

function getCijkl(C,i,j,k,l)
    function getvoigtindex(i,j)
        if i == j
            return i
        elseif (i == 2 && j==3) || (i == 3 && j==2)
            return 4
        elseif (i == 1 && j==3) || (i == 3 && j==1)
            return 5
        elseif (i == 1 && j==2) || (i == 2 && j==1)
            return 6
        end
    end
    m = getvoigtindex(i,j)
    n = getvoigtindex(k,l)
    return C[m,n]
end

function getCij(c,i,j)
    function gettensorindex(i)
        if (i >0) && (i < 4)
            return i, i
        elseif i == 4
            return 2, 3
        elseif i == 5
            return 1, 3
        elseif i == 6
            return 1, 2
        end
    end
    k, l = gettensorindex(i)
    m, n = gettensorindex(j)
    return c[k,l,m,n]
end

function VoigtMatrix2Tensor(C)
    cijkl=zeros((3,3,3,3));
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    cijkl[i,j,k,l] = getCijkl(C,i, j,k,l)
                end
            end
        end
    end
    return cijkl
end

function transform_tensor(c,R,i,j,k,l)
    c_new = 0.0
    for r in 1:3
        for s in 1:3
            for t in 1:3
                for u in 1:3
                    c_new +=R[i,r]*R[j,s]*R[k,t]*R[l,u]*c[r,s,t,u]
                end
            end
        end
    end
    return c_new
end

function getRmatrix(n)
    e3 = [n[1], n[2], n[3]]
    e1 = [n[3]/(n[1]*sqrt(1+(n[3]/n[1])^2)), 0, -1.0/sqrt(1+(n[3]/n[1])^2)]
    e2 = cross(e3,e1)
    R = [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]
    return R
end



function rotateTensor(cijkl, n::Array{Float64,1})
    function getRmatrix(n)
        e3 = [n[1], n[2], n[3]]
        e1 = [n[3]/(n[1]*sqrt(1+(n[3]/n[1])^2)), 0, -1.0/sqrt(1+(n[3]/n[1])^2)]
        e2 = cross(e3,e1)
        R = [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]
        return R
    end
    R=getRmatrix(n)
    return rotateTensor(cijkl,R)
end


function rotateTensor(cijkl, R::Array{Float64,2})
    cijkl_new=zeros((3,3,3,3));
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    cijkl_new[i,j,k,l] = transform_tensor(cijkl,R,i,j,k,l)
                end
            end
        end
    end
    return cijkl_new
end

function Tensor2VoigtMatrix(cijkl)
    C_new = zeros(6,6)
    for j in 1:6
        for i in 1:6
            C_new[i,j] = getCij(cijkl,i,j)
        end
    end
    return C_new
end

end
