#include <iostream>
#include "dmp/DmpContextual.hpp"
#include "dmp/DmpContextualOneStep.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#include "functionapproximators/MetaParametersLWPR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <algorithm> 

using namespace std;
using namespace Eigen;
using namespace DmpBbo;


int main(int argc, char **argv)
{   

    int N_DIMS = 1; // number of dimensions of the trajectory 
    int N_TASK_PARAMETERS = 1; // number of task parameters
    
    if(argc == 2 )
    {
        istringstream istr( argv[1] );
        istr >> N_DIMS;
    } 
    else if (argc == 3 )
    {
        istringstream istr(string(argv[1])+" "+string(argv[2]));
        istr >> N_DIMS;
        istr >> N_TASK_PARAMETERS;
    }
    else
    {
        cout << "Wrong number of parameters." << endl;
        cout << "Usage: " << string(argv[0]) << "[n_dim [n_params]]" << endl;
    }
   

    int N_INPUT_DIM = 1; // number of expected dimensions of each 
                         // function approximator  
                         // (one for each dimension)
    N_INPUT_DIM += N_TASK_PARAMETERS; // each function approximator has further 
                                      // dimensions related to the task parameters

    string log = "/tmp";
    string results = "data/results";
    string trajectories = "data/trajectories";

    assert(boost::filesystem::exists(trajectories) && "there's no directory called 'data/trajectories'");

    

    // number of basis functions on each expected dimension
    VectorXd NBF = VectorXd::Constant(N_INPUT_DIM, 1000);
    //NBF[0] = 50.0; // Need some more along time dimension; 
    
    // each dimension of the trajectory as a function approximator
    MetaParametersLWPR meta_parameters(N_INPUT_DIM, NBF);
    vector<FunctionApproximator*> function_approximators(N_DIMS);    
    
    // we use local weighted regression
    for (int d=0; d<N_DIMS; d++)
        function_approximators[d] =  
                    new FunctionApproximatorLWPR(&meta_parameters);

    // define the dmp
    DmpContextual* dmp = new DmpContextualOneStep(
            N_DIMS, function_approximators);

    // // function approximators can predict goals
    // for (int d=0; d<N_DIMS; d++)
    //     dmp->set_policy_parameter_function_goal(function_approximators[d]);
    // 
    //////////////////////////////////////////////////////////////// 
    // TRAJECTORIES
    const boost::regex trains( "^tl.*" );
    const boost::regex tests( "^tt.*" );
    const boost::regex res( "^rtt.*" );

    std::vector< std::string > train_files;
    std::vector< std::string > test_files;

    vector<Trajectory> trainset_trajectories;  
    vector<Trajectory> testset_trajectories;  

    // Default ctor yields past-the-end
    boost::filesystem::directory_iterator end_itr; 
    
    int N_TRAINSET = 0; // number of trajectories in the training set (to be read)
    int N_TESTSET = 0; // number of trajectories in the test set (to be read)


    // read trajectories from the "trajectories" dir
    for( boost::filesystem::directory_iterator i( trajectories ); i != end_itr; ++i )
    {
        if( !boost::filesystem::is_regular_file( i->status() ) ) continue;
        
        auto filename = i->path().filename().string();
        auto filepath = i->path().string();
        
        boost::smatch what;
        if( boost::regex_match( filename, what, trains ) )
        {          
            train_files.push_back(filepath);
            N_TRAINSET++;     
        }
        else if ( boost::regex_match( filename, what, tests ) )
        { 
            test_files.push_back(filepath);
            N_TESTSET++;
        }
    }


    std::sort(train_files.begin(), train_files.end());
    std::sort(test_files.begin(), test_files.end());

    for(auto &path: train_files)
    {
        Trajectory trj = Trajectory::readFromFile(path, N_TASK_PARAMETERS);
        trainset_trajectories.push_back(trj);
    }


    for(auto &path: test_files)
    {
        Trajectory trj = Trajectory::readFromFile(path, N_TASK_PARAMETERS);
        testset_trajectories.push_back(trj);
    }

    assert(N_TRAINSET>0 && "There's no trajectory here to learn");
    assert(N_TESTSET>0 && "There's no trajectory here to test");

    //////////////////////////////////////////////////////////////// 

    // collect params
    MatrixXd train_task_parameters(N_TRAINSET, N_TASK_PARAMETERS);
    for(int t=0; t<N_TRAINSET; t++)
        train_task_parameters.row(t) =
            trainset_trajectories[t].misc().row(0);
    

    MatrixXd testset_task_parameters(N_TESTSET, N_TASK_PARAMETERS);
    for(int t=0; t<N_TESTSET; t++)
        testset_task_parameters.row(t) =
            testset_trajectories[t].misc().row(0);

    // start training 
    bool overwrite = true; // if we can overvrite files in the results 
    dmp->train(trainset_trajectories, log, overwrite);
     
    // INTEGRATE THE CONTEXTUAL DMP FOR DIFFERENT TASK PARAMETERS
    
    vector<Trajectory> reproduced_trains(N_TRAINSET);
    vector<Trajectory> reproduced_trajectories(N_TESTSET);
    vector<MatrixXd> forcing_train_terms(N_TRAINSET);
    vector<MatrixXd> forcing_terms(N_TESTSET);
    
    // Integrate the DMP analytically 
    for (int i_train=0; i_train<N_TRAINSET; i_train++)
    {
        cout << "        " << (i_train+1) << "/" << N_TRAINSET << ":  ";
        for(int p=0; p< N_TASK_PARAMETERS; p++)
            cout << train_task_parameters(i_train, p) << " ";
        cout << endl;

        dmp->set_task_parameters(train_task_parameters.row(i_train));
        
        dmp->set_initial_state(trainset_trajectories[i_train].initial_y());
        //dmp->set_attractor_state(trainset_trajectories[i_train].final_y());
        
        dmp->analyticalSolution(
                trainset_trajectories[i_train].ts(),
                reproduced_trains[i_train],
                forcing_train_terms[i_train]);

        reproduced_trains[i_train].set_misc(
                trainset_trajectories[i_train].misc());

    }

   
    for (int i_test=0; i_test<N_TESTSET; i_test++)
    {
        cout << "        " << (i_test+1) << "/" << N_TESTSET << ":  ";
        for(int p=0; p< N_TASK_PARAMETERS; p++)
            cout << testset_task_parameters(i_test, p) << " ";
        cout << endl;
        
        dmp->set_initial_state(testset_trajectories[i_test].initial_y());
        //dmp->set_attractor_state(testset_trajectories[i_test].final_y());
       
        dmp->set_task_parameters(testset_task_parameters.row(i_test));
        dmp->analyticalSolution(
                testset_trajectories[i_test].ts(),
                reproduced_trajectories[i_test],
                forcing_terms[i_test]);

        reproduced_trajectories[i_test].set_misc(
                testset_trajectories[i_test].misc());

    }
     
    delete dmp;
    
    // reset results
    for( boost::filesystem::directory_iterator i( results ); i != end_itr; ++i )
    {
        if( !boost::filesystem::is_regular_file( i->status() ) ) continue;
        auto filename = i->path().filename().string();
        auto filepath = i->path().string();
        boost::smatch what;
        if( boost::regex_match( filename, what, res ) )
        { 
            boost::filesystem::remove(filepath);
        }
    }
   
    {
        
        bool overwrite = true;
 
        
        cout << "Saving reproduced trajectories to " << results << endl;
        for (int i_train=0; i_train<N_TRAINSET; i_train++)
        {
            stringstream stream;
            stream << "rtl" << setw(2) << setfill('0') << i_train;
            reproduced_trains[i_train].saveToFile(results,stream.str(),overwrite);
        }

        for (int i_test=0; i_test<N_TESTSET; i_test++)
        {
            stringstream stream;
            stream << "rtt" << setw(2) << setfill('0') << i_test;
            reproduced_trajectories[i_test].saveToFile(results,stream.str(),overwrite);
        }

 
    }



    return 0;
}
