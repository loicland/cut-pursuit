clear all;

%%%  parameters  %%%
%workDir = '~/Recherche/Optimization/Graph_quadratic_d1_l1/'
workDir = './'
dataset = 'Data/HugoData';
verbose = 1; % number of iterations between two progress messages

%%%  initialize and format data (single precision) %%%
%cd(workDir);
load(dataset, 'G', 'S', 'X', 'V');
%% format data for the optimization problem
% get the time instant with highest activity energy
E = sum(X.^2, 1);
[~, t] = max(E);
% extract relevent data
X = single(X);
y = X(:,t);
x0 = single(S(:,t)); % ground truth
if max(x0 <= 0) % all non zero are negative
    y = -y; % because we use positivity constraints
    x0 = -x0;
    X = -X;
end
Phi = single(G);
clear S G;
%% graph structure
[Eu, ~] = find(V' == 1);
[Ev, ~] = find(V' == -1);
% convert to C indices
Eu = int32(Eu - 1);
Ev = int32(Ev - 1);
[N, V] = size(Phi); % V is the number of vertices
E = length(Eu); % E is the number of edges

%%%  penalization parameters  %%%
% norms of columns of G;
% ColPhi = sqrt(sum(Phi.^2, 1))';
%% estimate noise on coefficients
% get single-coefficients 'normalized' pseudo-inverse
% CorPhiX = bsxfun(@rdivide, Phi'*X, ColPhi);
% get actual pseudo-inverse
InvPhiX = pinv(Phi)*X;
% estimate noise directly on time activity
Sigma = MAD_std_estimator(InvPhiX')';
% % estimate noise directly on high frequency time activity
% DifPhiX = abs(InvPhiX(:,2:2:end) - InvPhiX(:,1:2:end-1))/sqrt(2);
% Sigma = MAD_std_estimator(DifPhiX', true)';
% extract coefficients of 'normalized' pseudo-inverse at time of interest
x = InvPhiX(:,t);
%% d1 penalization
la_d1_min = 1e-3;
la_d1_max = 3;
nLa_d1 = 1000;
La = logspace(log10(single(la_d1_min)), log10(single(la_d1_max)), nLa_d1);
% get rough noise scaling of penalization coefficients
Mu = max(Sigma(Eu+1), Sigma(Ev+1));
% SURE
[SURE, VAR] = SURE_VAR_prox_graph_d1_mex(x, Sigma.^2, Mu, La, Eu, Ev, verbose);
[~, l] = min(SURE - VAR);
la_d1 = La(l);
%{
clf
subplot(2, 1, 1);
plot(La, [SURE VAR SURE-VAR], '.');
hold on;
plot(la_d1, SURE(l) - VAR(l), '*k')
title('\delta_1');
legend('SURE', 'VAR', 'SURE - VAR')
%}

%% l1 penalization
% SURE
[SURE, La] = SURE_prox_l12(abs(x), Sigma.^2, (x.*Sigma).^2, Sigma);
[~, l] = min(SURE);
la_l1 = La(l);
%{
subplot(2, 1, 2);
plot(La, SURE, '.');
hold on;
plot(la_l1, SURE(l), '*k')
title('l_1');
%}

%% take into account norms of columns of G
% scale penalizations according to time fluctuation of correlations
Sigma = MAD_std_estimator((Phi'*X)')';
La_d1 = (la_d1/2)*(Sigma(Eu+1) + Sigma(Ev+1));
La_l1 = la_l1*Sigma;
clear CorPhiX InvPhiX X Sigma Mu La ColPhi x SURE VAR;
