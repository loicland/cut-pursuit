%%%  penalizations  %%%
% La_d1_ = 100*ones(size(La_d1), class(y));
% La_l1_ = 50*ones(size(La_l1), class(y));
% La_l1_ = zeros(size(La_l1), class(y));
% La_l1_ = 0;
La_d1_ = La_d1;
La_l1_ = La_l1;
positivity = 0;

%%%  optimization parameters  %%%
%%  PFDR  %%
rho = 1.5;
condMin = 1e-1;
difRcd = 0;
difTol = 1e-4;
itMax = 1e4;
verbose = 100;

%%  CP  %%
CP_difTol = 1e-2;
CP_itMax = 10;
PFDR_rho = 1.5;
PFDR_condMin = 1e-3;
PFDR_difRcd = 0;
PFDR_difTol = 1e-5;
PFDR_itMax = 1e4;
PFDR_verbose = 1e3;

% %{
tic;
l = operator_norm_matrix(Phi);
[x, it, obj] = PFDR_graph_quadratic_d1_l1_mex(y, Phi, Eu, Ev, La_d1_, La_l1_, positivity, l, rho, condMin, difRcd, difTol, itMax, verbose);
t_pfdr = toc
it_pfdr = double(it);
obj_pfdr = obj(1:it_pfdr+1);
% dif_pfdr = dif(1:it_pfdr);
tim_pfdr = (0:it_pfdr)*t_pfdr/it_pfdr;
x_pfdr = x;
%}

%{
tic;
l = operator_norm_matrix(Phi);
[x, it, obj] = PFDR_graph_quadratic_d1_l1_AtA_mex(Phi'*y, Phi'*Phi, Eu, Ev, La_d1_, La_l1_, 0, l, rho, condMin, difRcd, difTol, 100, 10);
t_pfdrs = toc
it_pfdrs = double(it);
obj_pfdrs = obj(1:it_pfdrs+1) + sum(y.^2)/2;
% dif_pfdrs = dif(1:it_pfdrs);
tim_pfdrs = (0:it_pfdrs)*t_pfdrs/it_pfdrs;
x_pfdrs = x;
%}

%{
tic
[x, rx, it, tim, obj, dif] = CP_PFDR_graph_quadratic_d1_l1_mex(y, Phi, Eu, Ev, La_d1_, La_l1_, positivity, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, PFDR_verbose);
t_cp = toc
it_cp = double(it);
tim_cp = tim(1:it_cp+1);
obj_cp = obj(1:it_cp+1);
dif_cp = dif(1:it_cp);
x_cp = rx(x+1);
%}

%{
tic
[x, rx, it, tim, obj, dif] = CP_PFDR_graph_quadratic_d1_l1_AtA_mex(Phi'*y, Phi'*Phi, Eu, Ev, La_d1_, La_l1_, positivity, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, PFDR_verbose);
t_cps = toc
it_cps = double(it);
tim_cps = tim(1:it_cps+1);
obj_cps = obj(1:it_cps+1) + sum(y.^2)/2;
% dif_cps = dif(1:it_cps);
x_cps = rx(x+1);
%}

% %{
figure(1)
leg = {'PFDR', 'CP', 'CPs'};
cols = jet(length(leg));
% cols = [1 0 0; 0 0 1];
clf, i = 1;
% subplot(2, 1, 1);
semilogy(tim_pfdr,  obj_pfdr,  'color', cols(i,:), 'LineWidth', 2); i = i+1;
hold on
% semilogy(tim_pfdrs, obj_pfdrs, 'color', cols(i,:), 'LineWidth', 2); i = i+1;
semilogy(tim_cp,    obj_cp,    'color', cols(i,:), 'LineWidth', 2); i = i+1;
semilogy(tim_cps,   obj_cps,    'color', cols(i,:), 'LineWidth', 2); i = i+1;
legend(leg, 'Fontsize', 14);
xlabel('time (s)');
ylabel('objective');
axis tight
%}

% %{
top = max([x0; x_pfdr]);
figure(2)
leg = {'x_0', 'x_{PFDR}', 'x_{CP}'};
cols = [0 0 0; jet(length(leg)-1)];
clf, i = 1;
plot(x0 + top*i, 'color', cols(i,:), 'LineWidth', 2); i = i+1;
hold on;
plot(x_pfdr + top*i, 'color', cols(i,:), 'LineWidth', 2); i = i+1;
plot(x_cp + top*i, 'color', cols(i,:), 'LineWidth', 2); i = i+1;
legend(leg);
ylabel('x');
%}
