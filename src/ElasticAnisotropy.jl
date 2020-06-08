module ElasticAnisotropy

struct OrthoModel
    E_i::Float64
    E_a::Float64
    ν_i::Float64
    ν_ia::Float64
    G_a::Float64
end

function generateVoigtMatrix(X::OrthoModel)
    G_i = X.E_i / (2.0*(1. + X.ν_i))
    ν_ai = X.ν_ia * X.E_a / X.E_i
    D = ((1.0+X.ν_i)*(1.0-X.ν_i-2*X.ν_ia * X.ν_ai))/(X.E_i^2*X.E_a)
    a_ii = (1-X.ν_ia*ν_ai)/(X.E_i*X.E_a*D)
    a_ai = (1-X.ν_i^2)/(X.E_i^2*D)
    b_ii = (X.ν_i+X.ν_ia*ν_ai)/(X.E_i*X.E_a*D)
    b_ai = (X.ν_ia*(1+X.ν_i))/(X.E_i^2*D)
    c_ii = X.E_i/(2*(1+X.ν_i))
    c_ai = X.G_a
    C = [a_ii b_ii b_ai 0 0 0; b_ii a_ii b_ai 0 0 0; b_ai b_ai a_ai 0 0 0; 0 0 0 c_ai 0 0; 0 0 0 0 c_ai 0; 0 0 0 0 0 c_ii]
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

function rotateTensor(cijkl, v::Vector)
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


function rotateTensor(cijkl, R::Array)
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
        C_new[i,j] = getCij(cijkl_new,i,j)
    end
    return C_new
end

end

export  rotateTensor, Tensor2VoigtMatrix, VoigtMatrix2Tensor, generateVoigtMatrix
