#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <amdis/Marker.hpp>

#include <amdis/localoperators/StokesOperator.hpp>

#include <dune/grid/albertagrid.hh>
#include <cmath>
#include <typeinfo>
//#include <dune/alugrid/grid.hh>
//#include <dune/foamgrid/foamgrid.hh>

#include <amdis/project_active_surfaces/CHNSProb.h>

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

    auto grid = MeshCreator<Grid>("chMesh").create();
    //Dune::YaspGrid<2> grid{ {1.,2.}, {1,2}};

    //TODO Basis-Erzeugung in Klasse verschieben
    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());

    auto chnsBasis = composite(chBasis, stokesBasis);

    //CHNSProblem prob("CHNS", *grid, chnsBasis); //TODO: Figure out arguments
    CHNSProb prob("chns", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    AdaptInfo adaptInfo("adapt");
    prob.initBaseProblem(adaptInfo);

    AdaptInstationary adaptInst("adapt", prob, adaptInfo, prob, adaptInfo);
    adaptInst.adapt();

    return 0;
}