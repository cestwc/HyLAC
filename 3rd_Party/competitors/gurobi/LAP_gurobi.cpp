#include <gurobi_c++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <stdlib.h>
#include "include/config.h"
#include "include/Timer.h"
#include "include/cost_generator.h"

using namespace std;

double *read_normalcosts(double *C, int *Nad, const char *filepath)
{
    string s = filepath;
    ifstream myfile(s.c_str());
    if (!myfile)
    {
        std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
        exit(-1);
    }
    myfile >> Nad[0];
    int N = Nad[0];
    C = new double[N * N];
    for (int i = 0; i < N * N; i++)
    {
        myfile >> C[i];
    }
    myfile.close();
    return C;
}

int main(int argc, char **argv)
{
    Config config = parseArgs(argc, argv);
    printf("\033[0m");
    printf("Welcome ---------------------\n");
    printConfig(config);

    const int seed = config.seed;
    int N = config.user_n;

    double range = strtod(argv[2], nullptr);
    int N2 = N * N;

    typedef uint data;
    double time;
    Timer t;
    data *C = generate_cost<data>(config, seed);

    try
    {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        GRBVar *x = new GRBVar[N2];
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {

                // costx represents the coefficient of x variables in the objective function
                data costx = C[N * i + k];
                stringstream s;
                s << "X_" << i << "_" << k << endl;
                x[N * i + k] = model.addVar(0.0, 1.0, costx, GRB_CONTINUOUS, s.str());
            }
        }
        model.update();
        for (int k = 0; k < N; k++)
        {
            GRBLinExpr lhs = 0;
            for (int i = 0; i < N; i++)
            {
                lhs += x[N * i + k];
            }
            stringstream s;
            s << "XR_" << k << endl;
            model.addConstr(lhs == 1, s.str());
        }
        model.update();

        for (int i = 0; i < N; i++)
        {
            GRBLinExpr lhs = 0;
            for (int k = 0; k < N; k++)
            {
                lhs += x[N * i + k];
            }
            stringstream s;
            s << "XR1_" << i << endl;
            model.addConstr(lhs == 1, s.str());
        }
        model.update();
        auto start = t.elapsed();
        model.optimize();
        auto elapsed = t.elapsed() - start;

        cout << "Objective: " << model.getObjective().getValue() << endl;
        cout << "Time: " << elapsed << " s" << endl;
    }
    catch (GRBException e)
    {
        cout << "Error code = "
             << e.getErrorCode()
             << endl;
        cout << e.getMessage() << endl;
    }
    catch (...)
    {
        cout << "Exception during optimization"
             << endl;
    }

    return 0;
}