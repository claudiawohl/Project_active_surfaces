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
#include <dune/alugrid/grid.hh>
//#include <dune/foamgrid/foamgrid.hh>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//using Param = LagrangeBasis<Grid, 1, 1, 2, 2, 2, 1>;
//using Grid = Dune::YaspGrid<WORLDDIM>;

//auto Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//auto Grid = Dune::ALUGrid<GRIDDIM, WORLDDIM, Dune::simplex, Dune::nonconforming>();
//auto Grid = Dune::FoamGrid<GRIDDIM, WORLDDIM>();
//Dune::FoamGrid<GRIDDIM, WORLDDIM> Grid();

auto _0 = Dune::Indices::_0;
auto _1 = Dune::Indices::_1;

auto axisymmFunction = [](auto constant, double centre){
    return [constant, centre](auto const& x) {return constant/(x[0]-centre);};
};

int main(int argc, char** argv)
{
    Environment env(argc, argv);

    auto grid = MeshCreator<Grid>("chMesh").create();
    //Dune::YaspGrid<2> grid{ {1.,2.}, {1,2}};

    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());

    auto chnsBasis = composite(chBasis, stokesBasis);

    ProblemStat prob("ch", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    ProblemInstat probInstat("ch", prob);
    probInstat.initialize(INIT_UH_OLD);

    AdaptInfo adaptInfo("adapt");

    //FUNCTIONS from the previous time step
    auto invTau = std::ref(probInstat.invTau());
    auto phi = prob.solution(_0,0);
//    auto phiOld = probInstat.oldSolution(_0,0);  //replaced by phi
    auto gradPhi = gradientOf(phi);
    auto v = prob.solution(_1,_0);

    bool axisymmetric = Parameters::get<bool>("parameters->axisymmetric").value_or(false);

    double centre = 0.;
    if (axisymmetric) {centre = 0.;};

    auto upper_bound = [](auto const& x){
        return Dune::FieldVector<double,1>{1.};
    };
    auto lower_bound = [](auto const& x) {
        return Dune::FieldVector<double, 1>{0.};
    };
    auto phiProjected = max(min(evalAtQP(phi),upper_bound),lower_bound);
    //auto phiProjected = phi;

    //PATHS
    auto _phi =makeTreePath(_0,0);
    auto _mu =makeTreePath(_0,1);
    auto _v =makeTreePath(_1,_0);
    auto _p =makeTreePath(_1,1);

    //FIRST EQUATION
    //time derivative
    prob.addMatrixOperator(zot(invTau), _phi, _phi);
    prob.addVectorOperator(zot(phi * invTau), _phi);

    //Coupling term
    //auto opFconv = makeOperator(tag::test_gradtrial{}, v);
    //prob.addMatrixOperator(opFconv, _phi, _phi);

    //auto opFconv = makeOperator(tag::test_gradtrial{}, FieldVector<double, 2>{0, 0.98});
    //prob.addMatrixOperator(opFconv, _phi, _phi);

    auto opFconv = makeOperator(tag::test_trialvec{}, gradPhi, 5);
    prob.addMatrixOperator(opFconv, _phi, _v);                          //Coupling term

    //mobility term
    double m = Parameters::get<double>("parameters->mobility").value_or(0.4);
    //prob.addMatrixOperator(sot(m*pow<2>(phi)*pow<2>(FieldVector<double,1>{1.}-phi)), _phi, _mu);
    prob.addMatrixOperator(sot(m), _phi, _mu);

    if (axisymmetric)
    {
        auto opAddExtra = makeOperator(tag::test_partialtrial{0}, axisymmFunction(-m, centre), 3);
        prob.addMatrixOperator(opAddExtra, _phi, _mu);
    };

    //SECOND EQUATION
    prob.addMatrixOperator(zot(1.0), _mu, _mu);

    //TODO: Why is the multiplication with 3/sqrt(2) correct? Is it?
    double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*3*sqrt(2.);
    double a = Parameters::get<double>("parameters->a").value_or(1.0);
    double b = Parameters::get<double>("parameters->b").value_or(1.0);

    double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
    prob.addMatrixOperator(sot(-a*eps), _mu, _phi);

    if (axisymmetric)
    {
        auto opAddExtra = makeOperator(tag::test_partialtrial{0}, axisymmFunction(a*eps, centre), 3);
        prob.addMatrixOperator(opAddExtra, _mu, _phi);
    };

    //double Well potential: phi^2(1-phi)^2
    //TODO: Change expression as well to vector? - Not necessary - why not?
    auto opFimpl = zot(-b/eps * (FieldVector<double,1>{2.} + 12*phi*(phi - FieldVector<double,1>{1.})));
    prob.addMatrixOperator(opFimpl, _mu, _phi);

    auto opFexpl = zot(b/eps * pow<2>(phi)*(FieldVector<double,1>{6.} - 8*phi));
    prob.addVectorOperator(opFexpl, _mu);

    //THIRD EQUATION

    // define a constant fluid density
    double density_inner = 100.0;
    double density_outer = 1000.0;
    auto density = density_inner*phiProjected+(Dune::FieldVector<double,1>{1.}-phiProjected)*density_outer;

// <1/tau * u, v>
    auto opTime = makeOperator(tag::testvec_trialvec{},
                               density* invTau, 5);
    prob.addMatrixOperator(opTime, _v, _v);

// <1/tau * u^old, v>
    auto opTimeOld = makeOperator(tag::testvec{},
                                  density * invTau * prob.solution(_v), 5);
    prob.addVectorOperator(opTimeOld, _v);

    for (int i = 0; i < WORLDDIM; ++i) {
        // <(u^old * nabla)u_i, v_i>
        auto opNonlin = makeOperator(tag::test_gradtrial{},
                                     density * prob.solution(_v), 5);
        prob.addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }

   //define  a fluid viscosity
    double outer_visc = 10.;
    double inner_visc = 1.;
    auto viscosity = inner_visc*phiProjected+outer_visc*(FieldVector<double,1>{1.}-phiProjected);

    //Stokes Operator
    auto opStokes = makeOperator(tag::stokes{}, viscosity, 5);
    prob.addMatrixOperator(opStokes, _1, _1);

    if(axisymmetric){

        auto extra3D = [centre](auto const& x){
            return -1./(x[0]-centre);
        };

      auto opExtraX = makeOperator(tag::test_partialtrial{0}, viscosity*extra3D, 5);
      auto opExtraY = makeOperator(tag::test_partialtrial{1}, viscosity*extra3D, 5);
      auto opExtraGrad = makeOperator(tag::testvec_gradtrial{}, viscosity*extra3D, 5);

      prob.addMatrixOperator(opExtraX, makeTreePath(_1, _0, 0), makeTreePath(_1, _0, 0));
      prob.addMatrixOperator(opExtraY, makeTreePath(_1, _0, 1), makeTreePath(_1, _0, 0));
      prob.addMatrixOperator(opExtraGrad, _v, makeTreePath(_1, _0, 0));
    };

    auto opCoup = makeOperator(tag::testvec_trial{}, -sigma*gradPhi);
    prob.addMatrixOperator(opCoup, _v, _mu);

    //extern force (gravitational force)
    auto extForce = [](auto const& x) { return FieldVector<double,  WORLDDIM>{0.0, - 0.98};};
    auto opForce = makeOperator(tag::testvec {}, density*extForce, 5);
    prob.addVectorOperator(opForce, _v);


    //Initial Value and Boundary
    int ref_int  = Parameters::get<int>("refinement->interface").value_or(10);
    int ref_bulk = Parameters::get<int>("refinement->bulk").value_or(2);
    GridFunctionMarker marker("interface", prob.grid(),
                              invokeAtQP([ref_int, ref_bulk](double phi) {
                                  return phi > 0.05 && phi < 0.95 ? ref_int : ref_bulk;
                              }, phi));
    prob.addMarker(marker);

    /* Set old initial value
     double radius1 = Parameters::get<double>("parameters->radius1").value_or(0.15);
     double radius2 = Parameters::get<double>("parameters->radius2").value_or(0.25);
     for (int i = 0; i < 6; ++i)
     {
        phi << [eps,radius1,radius2](auto const& x)
        {
        using Math::sqr;
        return 0.5*(1 - std::tanh((radius1+radius2)*(std::sqrt(sqr(x[0]/radius1) + sqr(x[1]/radius2)) - 1.0)/(4*std::sqrt(2.0)*eps)));
        };
     prob.markElements(adaptInfo);
     prob.adaptGrid(adaptInfo);
     }
     */

    //set initial value for Benchmark
    double radius = Parameters::get<double>("parameters->radius").value_or(0.5);

    double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
    if (axisymmetric) {centerx = centre;};
    double centery = Parameters::get<double>("parameters->centery").value_or(0.);

    for (int i = 0; i < 10; ++i) {
        phi << [eps, radius, centerx, centery](auto const& x) {
            using Math::sqr;
            return 0.5*(1 - std::tanh((std::sqrt(sqr(x[0]-centerx) + sqr(x[1]-centery))- radius)/(std::sqrt(2.0)*eps)));
        };
        prob.markElements(adaptInfo);
        prob.adaptGrid(adaptInfo);
    }

    //boundary condition
    prob.addDirichletBC([](auto const& x) {return (x[1]<1.e-8 || x[1]>2-1.e-8);}, _v, _v, FieldVector<double, WORLDDIM>{0., 0.});
    prob.addDirichletBC([](auto const& x) {return (x[0]<1.e-8 || x[0]>1-1.e-8);}, makeTreePath(_1,_0,0), makeTreePath(_1,_0,0), 0.);
    //prob.addDirichletBC([](auto const& x) {return (x[0]<1.e-8 && x[1]<1.e-8);}, makeTreePath(_1,_1), makeTreePath(_1,_1), 0.); //set boundary for pressure - ensures regularity

    AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
    adapt.adapt();

    return 0;
}