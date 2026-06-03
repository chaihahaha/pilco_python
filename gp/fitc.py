# fitc.py
# *Summary:* Compute the FITC negative log marginal likelihood and its
# derivatives with respect to the inducing inputs (we don't compute the
# derivatives with respect to the GP hyper-parameters)
#
#   function [nml, dnml] = fitc(induce, gpmodel)
#
# *Input arguments:*
#
#   induce          matrix of inducing inputs                       [M x D x uE]
#                   M: number of inducing inputs
#                   E: either 1 (inducing inputs are shared across target dim.)
#                      or     E (different inducing inputs for each target dim.)
#   gpmodel         GP structure (dict)
#     .hyp          log-hyper-parameters                               [D+2 x E]
#     .inputs       training inputs                                    [N x D]
#     .targets      training targets                                   [N x E]
#     .noise (opt)  noise
#
# *Output arguments:*
#
#   nlml             negative log-marginal likelihood
#   dnlml            derivative of negative log-marginal likelihood wrt
#                    inducing inputs
#
# Adapted from Ed Snelson's SPGP code.
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-21

import numpy as np
from ..util.maha import maha


def fitc(induce, gpmodel):
    # ridge = 1e-06;                       % jitter to make matrix better conditioned
    ridge = 1e-06

    # [N D] = size(gpmodel.inputs); E = size(gpmodel.targets,2);
    N, D = gpmodel['inputs'].shape
    E = gpmodel['targets'].shape[1]

    # [M uD uE] = size(induce);
    if induce.ndim == 3:
        M, uD, uE = induce.shape
    else:
        M, uD = induce.shape
        uE = 1
        induce = induce.reshape(M, uD, 1)

    # if uD ~= D || (uE~=1 && uE ~= E); error('Wrong size of inducing inputs'); end
    if uD != D or (uE != 1 and uE != E):
        raise ValueError('Wrong size of inducing inputs')

    # nlml = 0; dfxb = zeros(M, D); dnlml = zeros(M, D, E); % zero and allocate outputs
    nlml = 0.0
    dnlml = np.zeros((M, D, E))

    # for j = 1:E
    for j in range(E):
        # if uE > 1; u = induce(:,:,j); else u = induce; end
        if uE > 1:
            u = induce[:, :, j]
        else:
            u = induce[:, :, 0]

        # b = exp(gpmodel.hyp(1:D,j));                                 % length-scales
        b = np.exp(gpmodel['hyp'][:D, j])
        # c = gpmodel.hyp(D+1,j);                                 % log signal std dev
        c = gpmodel['hyp'][D, j]
        # sig = exp(2.*gpmodel.hyp(D+2,j));                           % noise variance
        sig = np.exp(2.0 * gpmodel['hyp'][D + 1, j])

        # xb = bsxfun(@rdivide,u,b');                 % divide inducing by lengthscales
        xb = u / b
        # x = bsxfun(@rdivide,gpmodel.inputs,b');     % divide inputs by length-scales
        x = gpmodel['inputs'] / b
        # y = gpmodel.targets(:,j);                                  % training targets
        y = gpmodel['targets'][:, j:j + 1]  # (N, 1)

        # Kmm = exp(2*c-maha(xb,xb)/2) + ridge*eye(M);
        Kmm = np.exp(2 * c - maha(xb, xb) / 2) + ridge * np.eye(M)
        # Kmn = exp(2*c-maha(xb,x)/2);
        Kmn = np.exp(2 * c - maha(xb, x) / 2)

        # Check whether Kmm is no longer positive definite. If so, return
        # try
        #   L = chol(Kmm)';
        # catch
        #   nlml = Inf; dnlml = zeros(size(params));
        #   return;
        # end
        try:
            L = np.linalg.cholesky(Kmm)  # lower triangular, same as chol(Kmm)' in MATLAB
        except np.linalg.LinAlgError:
            nlml = np.inf
            dnlml = np.zeros(induce.shape)
            return nlml, dnlml

        # V = L\Kmn;                                               % inv(sqrt(Kmm))*Kmn
        V = np.linalg.solve(L, Kmn)

        # if isfield(gpmodel,'noise')
        #   Gamma = 1 + (exp(2*c)-sum(V.^2)'+gpmodel.noise(:,j))/sig;
        # else
        #   Gamma = 1 + (exp(2*c)-sum(V.^2)')/sig;      % Gamma = diag(Knn-Qnn)/sig + I
        # end
        if 'noise' in gpmodel:
            Gamma = 1 + (np.exp(2 * c) - np.sum(V ** 2, axis=0)[:, np.newaxis] + gpmodel['noise'][:, j:j + 1]) / sig
        else:
            Gamma = 1 + (np.exp(2 * c) - np.sum(V ** 2, axis=0)[:, np.newaxis]) / sig

        # V = bsxfun(@rdivide,V,sqrt(Gamma)');  % inv(sqrt(Kmm))*Kmn * inv(sqrt(Gamma))
        V = V / np.sqrt(Gamma).T
        # y = y./sqrt(Gamma);
        y = y / np.sqrt(Gamma)
        # Am = chol(sig*eye(M) + V*V')';        % chol(inv(sqrt(Kmm))*A*inv(sqrt(Kmm)))
        Am = np.linalg.cholesky(sig * np.eye(M) + V @ V.T)  # lower triangular
        # V*V' = inv(chol(Kmm)')*K*inv(diag(Gamma))*K'*inv(chol(Kmm)')'
        # Vy = V*y;
        Vy = V @ y
        # beta = Am\Vy;
        beta = np.linalg.solve(Am, Vy)

        # nlml = nlml + sum(log(diag(Am))) + (N-M)/2*log(sig) + sum(log(Gamma))/2 ...
        #        + (y'*y - beta'*beta)/2/sig + 0.5*N*log(2*pi);
        nlml += (np.sum(np.log(np.diag(Am)))
                 + (N - M) / 2 * np.log(sig)
                 + np.sum(np.log(Gamma)) / 2
                 + (float((y.T @ y).item()) - float((beta.T @ beta).item())) / 2 / sig
                 + 0.5 * N * np.log(2 * np.pi))

        # if nargout == 2               % ... and if requested, its partial derivatives
        # Compute derivatives always (caller expects them)

        # At = L*Am; iAt = At\eye(M);              % chol(sig*B) [Ed's thesis, p. 40]
        At = L @ Am
        iAt = np.linalg.solve(At, np.eye(M))
        # iA = iAt'*iAt;                                                 % inv(sig*B)
        iA = iAt.T @ iAt

        # iAmV = Am\V;                                                    % inv(Am)*V
        iAmV = np.linalg.solve(Am, V)
        # B1 = At'\(iAmV);
        B1 = np.linalg.solve(At.T, iAmV)
        # b1 = At'\beta;                                                  % b1 = B1*y
        b1 = np.linalg.solve(At.T, beta)

        # iLV = L'\V;                                 % inv(Kmm)*Kmn*inv(sqrt(Gamma))
        iLV = np.linalg.solve(L.T, V)
        # iL = L\eye(M);
        iL = np.linalg.solve(L, np.eye(M))
        # iKmm = iL'*iL;
        iKmm = iL.T @ iL

        # mu = ((Am'\beta)'*V)';
        # (Am'\beta) -> solve(Am.T, beta) shape (M,1)
        # (Am'\beta)' * V -> (1,M) @ (M,N) = (1,N)
        # transpose -> (N,1)
        mu = np.linalg.solve(Am.T, beta).T @ V
        mu = mu.T  # (N, 1)

        # bs = y.*(beta'*iAmV)'/sig - sum(iAmV.*iAmV)'/2 - (y.^2+mu.^2)/2/sig + 0.5;
        beta_iAmV = (beta.T @ iAmV).T  # (N, 1)
        bs = (y * beta_iAmV) / sig - np.sum(iAmV * iAmV, axis=0)[:, np.newaxis] / 2 - (y ** 2 + mu ** 2) / 2 / sig + 0.5

        # TT = iLV*(bsxfun(@times,iLV',bs));
        # iLV is (M,N), iLV' is (N,M), bs is (N,1)
        # bsxfun(@times,iLV',bs) -> (N,M) where each column of iLV' is multiplied by bs element-wise
        # In MATLAB: iLV' is N x M, bs is N x 1 -> bsxfun(@times,iLV',bs) multiplies each row of iLV' by bs(row)
        # Wait: bsxfun(@times, iLV', bs) with iLV' (N,M) and bs (N,1):
        #   bsxfun expands bs along columns: each column j of iLV' is multiplied by bs
        # So result is (N,M) where result[i,j] = iLV'[i,j] * bs[i]
        # In Python: iLV.T * bs  (broadcasting: (N,M) * (N,1) -> (N,M))
        # TT = iLV @ (iLV.T * bs)  -> (M,N) @ (N,M) = (M,M)
        TT = iLV @ (iLV.T * bs)

        # Kmn = bsxfun(@rdivide,Kmn,sqrt(Gamma)');                    % overwrite Kmn
        Kmn = Kmn / np.sqrt(Gamma).T

        # for i = 1:D                               % derivatives wrt inducing inputs
        dfxb = np.zeros((M, D))
        for i in range(D):
            # dsq_mm = bsxfun(@minus,xb(:,i),xb(:,i)').*Kmm;
            # xb(:,i) is (M,1), xb(:,i)' is (1,M)
            # bsxfun(@minus, xb(:,i), xb(:,i)') is (M,M) with entry (p,q) = xb(p,i) - xb(q,i)
            dsq_mm = (xb[:, i:i + 1] - xb[:, i:i + 1].T) * Kmm

            # dsq_mn = bsxfun(@minus,-xb(:,i),-x(:,i)').*Kmn;
            # -xb(:,i) is (M,1), -x(:,i)' is (1,N)
            # bsxfun(@minus, -xb(:,i), -x(:,i)') -> (M,N) entry (p,q) = -xb(p,i) - (-x(q,i)) = -(xb(p,i) - x(q,i))
            # Hmm wait, bsxfun(@minus, -xb(:,i), -x(:,i)') = -xb(:,i) - (-x(:,i)') = (x' - xb)
            # Each entry (p,q): -xb(p,i) - (-x(q,i)) = -xb(p,i) + x(q,i) = -(xb(p,i) - x(q,i))
            dsq_mn = (-xb[:, i:i + 1] - (-x[:, i:i + 1].T)) * Kmn

            # dGamma = -2/sig*dsq_mn.*iLV;
            dGamma = -2.0 / sig * dsq_mn * iLV

            # dfxb(:,i) = -b1.*(dsq_mn*(y-mu)/sig + dsq_mm*b1) + dGamma*bs ...
            #             + sum((iKmm - iA*sig).*dsq_mm,2) - 2/sig*sum(dsq_mm.*TT,2);
            dfxb[:, i] = (-b1.flatten() * (dsq_mn @ (y - mu) / sig + dsq_mm @ b1).flatten()
                          + (dGamma @ bs).flatten()
                          + np.sum((iKmm - iA * sig) * dsq_mm, axis=1)
                          - 2.0 / sig * np.sum(dsq_mm * TT, axis=1))

            # dsq_mn = dsq_mn.*B1;                                   % overwrite dsq_mn
            dsq_mn = dsq_mn * B1
            # dfxb(:,i) = dfxb(:,i) + sum(dsq_mn,2);
            dfxb[:, i] = dfxb[:, i] + np.sum(dsq_mn, axis=1)
            # dfxb(:,i) = dfxb(:,i)/b(i);
            dfxb[:, i] = dfxb[:, i] / b[i]

        # dnlml(:,:,j) = dfxb;
        dnlml[:, :, j] = dfxb

    # if 1 == uE; dnlml = sum(dnlml,3); end % combine derivatives if sharing inducing
    if uE == 1:
        dnlml = np.sum(dnlml, axis=2, keepdims=True)

    return nlml, dnlml
