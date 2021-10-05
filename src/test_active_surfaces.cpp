#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <amdis/localoperators/StokesOperator.hpp>
#include <amdis/Marker.hpp>

#include "amdis/extensions/CouplingBaseProblem.hpp"

#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/grid/albertagrid.hh>

#include <amdis/project_active_surfaces/CHNSProb.h>
#include <amdis/project_active_surfaces/ConcProb.h>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//using Param = LagrangeBasis<Grid, 1, 1, 2, 2, 2, 1>;
//using Grid = Dune::YaspGrid<WORLDDIM>;

//auto Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//auto Grid = Dune::ALUGrid<GRIDDIM, WORLDDIM, Dune::simplex, Dune::nonconforming>();
//auto Grid = Dune::FoamGrid<GRIDDIM, WORLDDIM>();
//Dune::FoamGrid<GRIDDIM, WORLDDIM> Grid();

int main(int argc, char** argv)
{
    Environment env(argc, argv);

    auto Basegrid = MeshCreator<Grid>("chMesh").create();
    AdaptiveGrid grid{*Basegrid};

    //TODO Basis-Erzeugung in Klasse verschieben
    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());
    auto chnsBasis = composite(chBasis, stokesBasis);

    CHNSProb probCHNS("chns", grid, chnsBasis);
    probCHNS.initialize(INIT_ALL);

    //Lagrange-Basis fixed!
    ConcProb concProb("conc", grid, probCHNS.phi());
    concProb.initialize(INIT_ALL);

    //TODO: Add coupling terms


    //TODO: Do INIT_ALL here and fix segmentation fault
    //CouplingBaseProblem prob{"prob", probCHNS, concProb};
    //prob.initialize(INIT_ALL);

    AdaptInfo adaptInfo("adapt");
    probCHNS.initBaseProblem(adaptInfo);
    probCHNS.initTimeInterface();
    concProb.initBaseProblem(adaptInfo);
    concProb.initTimeInterface();

    while (!(adaptInfo.reachedEndTime())) {
        adaptInfo.setTimestepIteration(0);
        adaptInfo.incTimestepNumber();
        adaptInfo.setTime(adaptInfo.time() + adaptInfo.timestep());

        if (adaptInfo.timestepNumber() == 0) {
            adaptInfo.setTime(adaptInfo.startTime());
            probCHNS.setTime(adaptInfo);
            probCHNS.solveInitialProblem(adaptInfo); // maybe initialAdaptInfo
            probCHNS.transferInitialSolution(adaptInfo);

            concProb.setTime(adaptInfo);
            concProb.solveInitialProblem(adaptInfo); // maybe initialAdaptInfo
            concProb.transferInitialSolution(adaptInfo);
        }

        std::cout << "time = " << adaptInfo.time() << ", end time = " << adaptInfo.endTime() << ", timestep = "
        << adaptInfo.timestep() << "\n";

        probCHNS.initTimestep(adaptInfo);
        probCHNS.beginIteration(adaptInfo);
        probCHNS.oneIteration(adaptInfo, FULL_ITERATION);
        probCHNS.endIteration(adaptInfo);
        probCHNS.closeTimestep(adaptInfo);

        concProb.initTimestep(adaptInfo);
        concProb.beginIteration(adaptInfo);
        concProb.oneIteration(adaptInfo, FULL_ITERATION);
        concProb.endIteration(adaptInfo);
        concProb.closeTimestep(adaptInfo);
    }


    return 0;
}