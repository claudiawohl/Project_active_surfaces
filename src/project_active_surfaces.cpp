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
#include <stdlib.h>
#include <typeinfo>
//#include <dune/alugrid/grid.hh>
//#include <dune/foamgrid/foamgrid.hh>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;

//TODO: Decouple System with Amdis-extensions
//TODO: Find correct ansatz polynomial degree

//TODO: Change grid
//auto Grid = Dune::ALUGrid<GRIDDIM, WORLDDIM, Dune::simplex, Dune::nonconforming>();
//auto Grid = Dune::FoamGrid<GRIDDIM, WORLDDIM>();

/// introduce index short cuts
auto _0 = Dune::Indices::_0;
auto _1 = Dune::Indices::_1;
auto _2 = Dune::Indices::_2;


template <typename T> std::string type_name();

/** \file
 *  \brief This is the full model
 *  \author Claudia Wohlgemuth
 */
int main(int argc, char** argv)
{
    Environment env(argc, argv);

    auto grid = MeshCreator<Grid>("chMesh").create();

    /// create basis tree
    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());
    auto concentrBasis = lagrange<2>();

    auto chnsBasis = composite(chBasis, stokesBasis, concentrBasis);

    /// create the problem enviroment
    ProblemStat prob("ch", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    ProblemInstat probInstat("ch", prob);
    probInstat.initialize(INIT_UH_OLD);

    AdaptInfo adaptInfo("adapt");

    bool axisymmetric = Parameters::get<bool>("parameters->axisymmetric").value_or(false);

    double centre = 0.;
    if (axisymmetric) {centre = 0.;};

    auto axisymmFunction = [centre](auto constant){
        return [constant, centre](auto const& x) {return constant/(x[0]-centre);};
    };

    /// FUNCTIONS from the previous time step
    auto invTau = std::ref(probInstat.invTau());
    auto phi = prob.solution(_0,0);
    auto gradPhi = gradientOf(phi);
    auto v = prob.solution(_1,_0);
    auto c = prob.solution(_2);
    auto phiProjected = max(min(evalAtQP(phi),Dune::FieldVector<double,1>{1.}), Dune::FieldVector<double, 1>{0.});

    /// define path shortcuts
    auto _phi =makeTreePath(_0,0);
    auto _mu =makeTreePath(_0,1);
    auto _v =makeTreePath(_1,_0);
    auto _p =makeTreePath(_1,_1);
    auto _c =makeTreePath(_2);

    /** \description The first to equations are part of the Cahn-Hilliard model
      \f(  \begin{cases}
       \partial_t \phi + v \cdot \nabla \phi \hspace{1.2cm} &= \nabla \cdot (M\nabla \mu)\\
       \mu &= \sigma \left(\varepsilon^{-1} W'(\phi)+\varepsilon\Delta \phi \right)
      \end{cases} \f).

      \$ \phi \in [0,1]\$... phase field
      \$ \mu \$... chemical potential
      \$ W = \phi^2(1-\phi)^2\$... Double Well Potential
      \$ \varepsilon \$ transition width
      \$ \sigma = \tilde \sigma + f(c)\$... surface tension

     We implement them in their weak formultation.
    **/

    /// FIRST EQUATION
    /**\f(
      (\partial_t \phi^m, \varphi_1) + (v^{m-1}\cdot \nabla \phi^m, \varphi_1) + (M\nabla \mu^m, \nabla \varphi_1)&=0
       \f)**/
    ///time derivative
    prob.addMatrixOperator(zot(invTau), _phi, _phi);
    prob.addVectorOperator(zot(phi * invTau), _phi);

    //Coupling term - convection term
    //auto opFconv = makeOperator(tag::test_gradtrial{}, FieldVector<double, 2>{0, 0.98}); //Test
    //prob.addMatrixOperator(opFconv, _phi, _phi);                                         //Test

    ///convection term (coupling)
    auto opFconv = makeOperator(tag::test_trialvec{}, gradPhi, 5);
    prob.addMatrixOperator(opFconv, _phi, _v);

    ///mobility term
    double m = Parameters::get<double>("parameters->mobility").value_or(0.4);
  //prob.addMatrixOperator(sot(m*pow<2>(phi)*pow<2>(FieldVector<double,1>{1.}-phi)), _phi, _mu);
    prob.addMatrixOperator(sot(m), _phi, _mu);

    if (axisymmetric)
    {
        auto opAddExtra = makeOperator(tag::test_partialtrial{0}, axisymmFunction(-m), 3);
        prob.addMatrixOperator(opAddExtra, _phi, _mu);
    };

    ///SECOND EQUATION
    /**\f(
     * (\mu^m, \varphi_2)-\sigma \varepsilon(\nabla \phi^m,\nabla \varphi_2)- \varepsilon^{-1}(W'(\phi^m),\varphi_2) &=0
     * \f)
     */
    double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*sqrt(2.)*3.;
    double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);

    ///mu
    prob.addMatrixOperator(zot(1.0), _mu, _mu);
    ///Laplace phi
    prob.addMatrixOperator(sot(-eps), _mu, _phi);

    if (axisymmetric)
    {
        auto opAddExtra = makeOperator(tag::test_partialtrial{0}, axisymmFunction(eps), 3);
        prob.addMatrixOperator(opAddExtra, _mu, _phi);
    };

    ///double Well potential: phi^2(1-phi)^2
    auto opFimpl = zot(-1./eps * (2. + 12*phi*(phi - 1.)));
    prob.addMatrixOperator(opFimpl, _mu, _phi);
    auto opFexpl = zot(1./eps * pow<2>(phi)*(6. - 8*phi));
    prob.addVectorOperator(opFexpl, _mu);

    /** The second two equations descrive the (Navier-)Stokes model for fluids
     * \f(
    \begin{cases}
    \rho(\phi) \left(\partial_t v + (v\cdot \nabla)v\right)
    &=-\nabla p + \nabla \cdot \left(\nu(\phi)(\nabla v+\nabla v^T)\right)+\sigma \mu\nabla\phi + f\\
    \nabla \cdot v &= 0
    \end{cases}
     * \f)
     *
     * with density \$rho\$ and velocity \$\nu\$.
     *
     * We set the density \$rho=0\$ to receive the Stokes model which will be used over the testing case.
     *
     * We implement them in their weak formulation.
     */

    ///THIRD EQUATION
    /**
     \begin{cases}
      \frac{1}{\Delta t}(\rho(\phi^{m-1}) v^m, \varphi_3) - (\rho(v^m\cdot \nabla)v^m,\varphi_3)&\\+ (\nu(\phi^{m-1}) \nabla v^m, \nabla \varphi_3)-(p^m, \nabla \varphi_3)-(\mu^m\nabla\phi^m, \varphi_3)&=(f,\varphi_3) + \frac{1}{\Delta t}(\rho(\phi^{m-1}) v^{m-1}, \varphi_3) \\
      (\nabla v^m, \varphi_4) &=0.
     \end{cases}
     **/

    ///Stokes equation
    // define  a fluid viscosity
    double outer_visc = Parameters::get<double>("parameters->outerviscosity").value_or(10.);
    double inner_visc = Parameters::get<double>("parameters->innerviscosity").value_or(1.);

    auto viscosity = inner_visc*phiProjected+outer_visc*(1.-phiProjected);

    //Stokes Operator
    auto opStokes = makeOperator(tag::stokes{}, viscosity, 5);
    prob.addMatrixOperator(opStokes, _1, _1);

    if(axisymmetric){

        auto opExtraX = makeOperator(tag::test_partialtrial{0}, viscosity*axisymmFunction(-1.), 5);
        auto opExtraY = makeOperator(tag::test_partialtrial{1}, viscosity*axisymmFunction(-1.), 5);
        auto opExtraGrad = makeOperator(tag::testvec_gradtrial{}, viscosity*axisymmFunction(-1.), 5);

        prob.addMatrixOperator(opExtraX, makeTreePath(_1, _0, 0), makeTreePath(_1, _0, 0));
        prob.addMatrixOperator(opExtraY, makeTreePath(_1, _0, 1), makeTreePath(_1, _0, 0));
        prob.addMatrixOperator(opExtraGrad, _v, makeTreePath(_1, _0, 0));
    };

if (true) {
    // define a constant fluid density
    double density_inner = Parameters::get<double>("parameters->innerdensity").value_or(100.);
    double density_outer = Parameters::get<double>("parameters->outerdensity").value_or(1000.);
    auto density = density_inner * phiProjected + (1. - phiProjected) * density_outer;

    ///time derivative
// <1/tau * u, v>
    auto opTime = makeOperator(tag::testvec_trialvec{},
                               density * invTau, 5);
    prob.addMatrixOperator(opTime, _v, _v);

// <1/tau * u^old, v>
    auto opTimeOld = makeOperator(tag::testvec{},
                                  density * invTau * prob.solution(_v), 5);
    prob.addVectorOperator(opTimeOld, _v);

    ///material derivative part
    for (int i = 0; i < WORLDDIM; ++i) {
        // <(u^old * nabla)u_i, v_i>
        auto opNonlin = makeOperator(tag::test_gradtrial{},
                                     density * prob.solution(_v), 5);
        prob.addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }

    ///extern force (gravitational force)
    auto extForce = [](auto const& x) { return FieldVector<double,  WORLDDIM>{0.0, - 0.98};};
    auto opForce = makeOperator(tag::testvec {}, density*extForce, 5);
    prob.addVectorOperator(opForce, _v);

}

    ///coupling term
    auto opCoup = makeOperator(tag::testvec_trial{}, -sigma*gradPhi);
    prob.addMatrixOperator(opCoup, _v, _mu);
/*
    for (int i=0; i<WORLDDIM; i++){
        auto partphi = partialDerivativeOf(phi,i);
        prob.addMatrixOperator(zot(-sigma*partphi), makeTreePath(_1,_0, i), _mu);
    }
*/
    /**
     * We now introduce the concentration equation.
     * \f(
     *  \partial_t c + \nabla_\Gamma \cdot (c v) - D \Delta_\Gamma c +\tau_D c = \tau_D
     * \f)
     *
     * where the coupling with the CHNS-system happens in the surface tension parameter
     * \$\sigma = \tilde \sigma + f(c)\$ with \$f(c) = \frac{2c^2}{c_0^2+c^2}\$.
     **/
    ///FIFTH EQUATION
    auto absGradPhi = two_norm(gradPhi) + 1e-6;  //tested it - seems to work: changes with time
    auto normal_vec = gradPhi/absGradPhi;
    auto Id=1.;
    auto NxN =outer(normal_vec,normal_vec);
    auto gradC = gradientOf(c);

    //time derivative
    prob.addMatrixOperator(zot(invTau*absGradPhi, 5), _c, _c);
    prob.addVectorOperator(zot(absGradPhi*invTau*c, 5), _c);

    //nonlinearity (time derivative)
    auto opConcVelTime = makeOperator(tag::test_gradtrial{}, v*absGradPhi,5);
    prob.addMatrixOperator(opConcVelTime, _c, _c);

    //Laplace term
    auto opConcLaplacian1 = makeOperator(tag::gradtest_gradtrial {}, absGradPhi * Id,5);
    prob.addMatrixOperator(opConcLaplacian1, _c, _c);
    auto opConcLaplacian2 = makeOperator(tag::gradtest_gradtrial {}, - absGradPhi * NxN,5);
    prob.addMatrixOperator(opConcLaplacian2, _c, _c);

    if(axisymmetric){
        auto PGradC_r = (FieldVector<double, 2>{1.,0.}-FieldVector<double, 2>{1.,0.}*NxN)*gradC;
        prob.addMatrixOperator(zot(PGradC_r*axisymmFunction(1.), 5),_c, _c);
    }

    //Convection term with Projection operator
    prob.addMatrixOperator(zot(absGradPhi* divergenceOf(v), 5), _c, _c);
    prob.addMatrixOperator(zot(-absGradPhi*normal_vec *gradientOf(v)*normal_vec, 5), _c, _c);

    if(axisymmetric){
        auto v_r = prob.solution(makeTreePath(_1, _0, 1));
        prob.addMatrixOperator(zot(v_r*axisymmFunction(1.), 5),_c, _c);
    }

    /*
    //Linear term
    double const0 = 10.;
    double const1 = 1.;
    prob.addMatrixOperator(zot(absGradPhi*const0, 5), _c, _c);
    prob.addVectorOperator(zot(absGradPhi*const0*const1, 5), _c);
     */

    //coupling term (in equation 3) surface tension
    double constPe = 100.;
    double c0 = 1.;

   /*  with projection operator: Which sign???
    auto opCoupC1 = makeOperator(tag::testvec {}, constPe*(constant2 + 2*Math::sqr(c)/(Math::sqr(c0)+ Math::sqr(c))), 5);
    prob.addVectorOperator(opCoupC1, _v);
    auto opCoupC2 = makeOperator(tag::testvec {}, (-1.)*constPe*(constant2 + 2*Math::sqr(c)/(Math::sqr(c0)+ Math::sqr(c)))*NxN, 5);
    prob.addVectorOperator(opCoupC2, _v);*/


    //Hill function term
    //auto gradf = gradientOf(constPe*(constant2 + 2*Math::sqr(c)/(Math::sqr(c0)+ Math::sqr(c))));
    auto opCoupC = makeOperator(tag::testvec_trial {}, (-1.)*sqrt(2.)*3.*constPe*(2*Math::sqr(c)/(Math::sqr(c0)+ Math::sqr(c)))*gradPhi, 5);
    prob.addMatrixOperator(opCoupC, _v, _mu);
    //Switched sign????
    auto opCoupC1 = makeOperator(tag::testvec {}, absGradPhi*(-constPe*4*Math::sqr(c0)*c/Math::sqr(Math::sqr(c0)+ Math::sqr(c))*gradC), 5);
    auto opCoupC2 = makeOperator(tag::testvec {}, absGradPhi*(constPe*4*Math::sqr(c0)*c/Math::sqr(Math::sqr(c0)+ Math::sqr(c))*gradC*NxN), 5);
    prob.addVectorOperator(opCoupC1, _v);
    prob.addVectorOperator(opCoupC2, _v);


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
    double centery = Parameters::get<double>("parameters->centery").value_or(0.);

    for (int i = 0; i < 10; ++i) {
        phi << [eps, radius, centerx, centery](auto const& x) {
            using Math::sqr;
            return 0.5*(1 - std::tanh((std::sqrt(sqr(x[0]-centerx) + sqr(x[1]-centery))- radius)/(std::sqrt(2.0)*eps)));
        };
        prob.markElements(adaptInfo);
        prob.adaptGrid(adaptInfo);
    }

    c << [](auto const& x){
        if (x[0]<0.5){return 0.5;};
        return 1.0;
    };

    //boundary condition
    prob.addDirichletBC([](auto const& x) {return (x[1]<1.e-8 || x[1]>2-1.e-8);}, _v, _v, FieldVector<double, WORLDDIM>{0., 0.});
    prob.addDirichletBC([](auto const& x) {return (x[0]<1.e-8 || x[0]>1-1.e-8);}, makeTreePath(_1,_0,0), makeTreePath(_1,_0,0), 0.);

    AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
    adapt.adapt();

    return 0;
}