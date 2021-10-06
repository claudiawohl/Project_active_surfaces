#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <amdis/localoperators/StokesOperator.hpp>
#include <amdis/Marker.hpp>

#include "amdis/extensions/CouplingBaseProblem.hpp"

#include <dune/grid/io/file/vtk/vtkwriter.hh>
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

    if (true){
        ///Coupling terms
        auto v = probCHNS.v();

        auto c = concProb.solution();
        auto gradC = gradientOf(c);
        auto gradPhi = concProb.gradPhi();
        auto absGradPhi = concProb.absGradPhi();
        auto normal_vec = concProb.normal_vec();
        auto NxN = concProb.NxN();

        // <v grad(c), psi>
        concProb.problem().addMatrixOperator(fot(v, tag::grad_test {}));
        // <c grad_(v), psi>
        concProb.problem().addMatrixOperator(zot(absGradPhi* divergenceOf(v), 5));
        concProb.problem().addMatrixOperator(zot(-absGradPhi*normal_vec *gradientOf(v)*normal_vec, 5));

        bool axisymmetric = false;//TODO:
        if(axisymmetric){
            //auto v_r = concProb.problem().solution(makeTreePath(Dune::Indices::_1, Dune::Indices::_0, 1));
            //concProb.problem().addMatrixOperator(zot(v_r*X(0), 5));
        }

        ///Hill-function terms
        double constPe = 1.;
        double c0 = 1.;

        auto opCoupC = makeOperator(tag::testvec_trial {}, (-1.)*sqrt(2.)*3.*constPe*(2*Math::sqr(c)/(Math::sqr(c0)+ Math::sqr(c)))*gradPhi, 5);
        probCHNS.problem().addMatrixOperator(opCoupC, probCHNS._v, probCHNS._mu);
        //Switched sign????
        auto opCoupC1 = makeOperator(tag::testvec {}, absGradPhi*(-constPe*4*Math::sqr(c0)*c/Math::sqr(Math::sqr(c0)+ Math::sqr(c))*gradC), 5);
        auto opCoupC2 = makeOperator(tag::testvec {}, absGradPhi*(constPe*4*Math::sqr(c0)*c/Math::sqr(Math::sqr(c0)+ Math::sqr(c))*gradC*NxN), 5);
        probCHNS.problem().addVectorOperator(opCoupC1, probCHNS._v);
        probCHNS.problem().addVectorOperator(opCoupC2, probCHNS._v);
    }

    AdaptInfo adaptInfo("adapt");
    probCHNS.initBaseProblem(adaptInfo);
    probCHNS.initTimeInterface();
    concProb.initBaseProblem(adaptInfo);
    concProb.initTimeInterface();

    if (adaptInfo.timestepNumber() == 0) {
        adaptInfo.setTime(adaptInfo.startTime());

        probCHNS.setTime(adaptInfo);
        probCHNS.solveInitialProblem(adaptInfo); // maybe initialAdaptInfo
        probCHNS.transferInitialSolution(adaptInfo);

        concProb.setTime(adaptInfo);
        concProb.solveInitialProblem(adaptInfo); // maybe initialAdaptInfo
        concProb.transferInitialSolution(adaptInfo);
    }

    //ToDo. Write out phi()
    // VtkWriter phiWriter{Grid::leafGridView(), Dune::VTK::nonconforming};

    while (!(adaptInfo.reachedEndTime())) {
        adaptInfo.setTimestepIteration(0);
        adaptInfo.incTimestepNumber();
        adaptInfo.setTime(adaptInfo.time() + adaptInfo.timestep());

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