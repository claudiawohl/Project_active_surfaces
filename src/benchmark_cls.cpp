//
// Created by claudia on 29.09.2021.
//
#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>

#include <dune/grid/albertagrid.hh>

#include <dune/functions/functionspacebases/compositebasis.hh>
#include <amdis/LocalOperators.hpp>
#include <amdis/localoperators/StokesOperator.hpp>
#include <amdis/Marker.hpp>

#include <amdis/project_active_surfaces/CHNSProb.h>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;

int main(int argc, char** argv)
{
    Environment env(argc, argv);

    auto grid = MeshCreator<Grid>("chMesh").create();

    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());

    auto chnsBasis = composite(chBasis, stokesBasis);

    CHNSProb prob("chns", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    AdaptInfo adaptInfo("adapt");
    prob.initBaseProblem(adaptInfo);

    AdaptInstationary adaptInst("adapt", prob, adaptInfo, prob, adaptInfo);
    adaptInst.adapt();

    return 0;
}