tmpDir = pwd;
cd(fileparts(which('compile_graph_quadratic_d1_l1_mex.m')));
CXXFLAGScat = '-fopenmp -DNDEBUG';
LDFLAGScat = '-fopenmp';
try
    % %{
    depList = {'PFDR_graph_quadratic_d1_l1'};
    compile_Cpp_mex('PFDR_graph_quadratic_d1_l1',     depList, CXXFLAGScat, LDFLAGScat, true);
    compile_Cpp_mex('PFDR_graph_quadratic_d1_l1_AtA', depList, CXXFLAGScat, LDFLAGScat, true);
    compile_Cpp_mex('PFDR_graph_l22_d1_l1', depList, CXXFLAGScat, LDFLAGScat, true);
    % depList = {'PGFB_graph_quadratic_d1_l1'};
    % compile_Cpp_mex('PGFB_graph_quadratic_d1_l1',     depList, CXXFLAGScat, LDFLAGScat, true);
    % compile_Cpp_mex('PGFB_graph_quadratic_d1_l1_AtA', depList, CXXFLAGScat, LDFLAGScat, true);
    % compile_Cpp_mex('PGFB_graph_quadratic_d1_l1_pos', depList, CXXFLAGScat, LDFLAGScat, true);
    %}
    % %{
    depList = {'CP_PFDR_graph_quadratic_d1_l1', 'PFDR_graph_quadratic_d1_l1', 'graph', 'maxflow', 'operator_norm_matrix'};
    compile_Cpp_mex('CP_PFDR_graph_quadratic_d1_l1',     depList, CXXFLAGScat, LDFLAGScat, true);
    compile_Cpp_mex('CP_PFDR_graph_quadratic_d1_l1_AtA', depList, CXXFLAGScat, LDFLAGScat, true);
    compile_Cpp_mex('CP_PFDR_graph_l22_d1_l1', depList, CXXFLAGScat, LDFLAGScat, true);
    %}
catch
	cd(tmpDir);
	rethrow(lasterror);
end
cd(tmpDir);
