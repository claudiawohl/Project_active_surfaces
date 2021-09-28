//
// Created by claudia on 18.09.2021.
//
#include <amdis/extensions/BaseProblem.hpp>
#include <amdis/localoperators/StokesOperator.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/GridFunctions.hpp>
#include <dune/functions/functionspacebases/compositebasis.hh>

namespace AMDiS{

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
    template <class Grid>
    void CHNSProblem<Grid>::fillCahnHilliard(AdaptInfo& adaptInfo) {

        bool axisymmetric = false;

        /// introduce index short cuts
        auto _0 = Dune::Indices::_0;
        auto _1 = Dune::Indices::_1;
        auto _2 = Dune::Indices::_2;

        /// define path shortcuts
        auto _phi =makeTreePath(_0,0);
        auto _mu =makeTreePath(_0,1);
        auto _v =makeTreePath(_1,_0);
        auto _p =makeTreePath(_1,_1);
        auto _c =makeTreePath(_2);

        /// FUNCTIONS from the previous time step
        auto invTau = std::ref(problem().invTau());
        auto phi = problem().solution(_phi);
        auto gradPhi = gradientOf(phi);
        auto v = problem().solution(_v);
        auto c = problem().solution(_c);
        auto phiProjected = max(min(evalAtQP(phi),Dune::FieldVector<double,1>{1.}), Dune::FieldVector<double, 1>{0.});


        /// FIRST EQUATION
        /**\f(
          (\partial_t \phi^m, \varphi_1) + (v^{m-1}\cdot \nabla \phi^m, \varphi_1) + (M\nabla \mu^m, \nabla \varphi_1)&=0
           \f)**/
        ///time derivative
        problem().addMatrixOperator(zot(invTau), _phi, _phi); //TODO: public class member.
        problem().addVectorOperator(zot(phi * invTau), _phi);

        //Coupling term - convection term
        //auto opFconv = makeOperator(tag::test_gradtrial{}, FieldVector<double, 2>{0, 0.98}); //Test
        //problem().addMatrixOperator(opFconv, _phi, _phi);                                         //Test

        ///convection term (coupling)
        auto opFconv = makeOperator(tag::test_trialvec{}, gradPhi, 5);
        problem().addMatrixOperator(opFconv, _phi, _v);

        ///mobility term
        double m = Parameters::get<double>("parameters->mobility").value_or(0.4); //TODO derive in main()
        //problem().addMatrixOperator(sot(m*pow<2>(phi)*pow<2>(FieldVector<double,1>{1.}-phi)), _phi, _mu);
        problem().addMatrixOperator(sot(m), _phi, _mu);

        if (axisymmetric)
        {
            auto opLaplaceMuAxi = makeOperator(tag::test_partialtrial{0}, -m/X(0), 3);
            problem().addMatrixOperator(opLaplaceMuAxi, _phi, _mu);
        };

        ///SECOND EQUATION
        /**\f(
         * (\mu^m, \varphi_2)-\sigma \varepsilon(\nabla \phi^m,\nabla \varphi_2)- \varepsilon^{-1}(W'(\phi^m),\varphi_2) &=0
         * \f)
         */
        double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*sqrt(2.)*3.; //TODO: derive in main()
        double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02); //TODO: derive in main()

        ///mu
        problem().addMatrixOperator(zot(1.0), _mu, _mu);
        ///Laplace phi
        problem().addMatrixOperator(sot(-eps), _mu, _phi);

        if (axisymmetric)
        {
            auto opLaplacePhiAxi= makeOperator(tag::test_partialtrial{0}, eps/X(0), 3);
            problem().addMatrixOperator(opLaplacePhiAxi, _mu, _phi);
        };

        ///double Well potential: phi^2(1-phi)^2
        auto opFimpl = zot(-1./eps * (2. + 12*phi*(phi - 1.)));
        problem().addMatrixOperator(opFimpl, _mu, _phi);
        auto opFexpl = zot(1./eps * pow<2>(phi)*(6. - 8*phi));
        problem().addVectorOperator(opFexpl, _mu);
        return;
    }

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
    template <class Grid>
    void CHNSProblem<Grid>::fillNavierStokes(AdaptInfo& adaptInfo) {

        bool axisymmetric = false;

        /// introduce index short cuts
        auto _0 = Dune::Indices::_0;
        auto _1 = Dune::Indices::_1;

        /// define path shortcuts
        auto _phi =makeTreePath(_0,0);
        auto _mu =makeTreePath(_0,1);
        auto _v =makeTreePath(_1,_0);
        auto _p =makeTreePath(_1,_1);

        /// FUNCTIONS from the previous time step
        auto invTau = std::ref(this->problem().invTau());
        auto phi = this->problem().solution(_phi);
        auto gradPhi = gradientOf(phi);
        auto v = this->problem().solution(_v);
        auto phiProjected = max(min(evalAtQP(phi),Dune::FieldVector<double,1>{1.}), Dune::FieldVector<double, 1>{0.});


        //TODO
        double sigma = 24.5;


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
        this->problem().addMatrixOperator(opStokes, _1, _1);

        if(axisymmetric){
            auto opExtraX = makeOperator(tag::test_partialtrial{0}, -viscosity/X(0), 5);
            auto opExtraY = makeOperator(tag::test_partialtrial{1}, -viscosity/X(0), 5);

            this->problem().addMatrixOperator(opExtraX, makeTreePath(_1, _0, 0), makeTreePath(_1, _0, 0));
            this->problem().addMatrixOperator(opExtraX, makeTreePath(_1, _0, 0), makeTreePath(_1, _0, 0));
            this->problem().addMatrixOperator(opExtraY, makeTreePath(_1, _0, 1), makeTreePath(_1, _0, 0));
            this->problem().addMatrixOperator(opExtraX, makeTreePath(_1, _0, 1), makeTreePath(_1, _0, 1));

            auto laplaceVAxi2 = makeOperator(tag::test_trial{},  2*viscosity/ (X(0) * X(0)), 5);
            this->problem().addMatrixOperator(laplaceVAxi2,makeTreePath(_1, _0, 0), makeTreePath(_1, _0, 0));

            auto opExtra = makeOperator(tag::test_trial{}, 1./X(0), 5);
            this->problem().addMatrixOperator(opExtra, makeTreePath(_1, _1), makeTreePath(_1, _0, 0));
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
            this->problem().addMatrixOperator(opTime, _v, _v);

            // <1/tau * u^old, v>
            auto opTimeOld = makeOperator(tag::testvec{},
                                          density * invTau * this->problem().solution(_v), 5);
            this->problem().addVectorOperator(opTimeOld, _v);

            ///material derivative part
            for (int i = 0; i < WORLDDIM; ++i) {
                // <(u^old * nabla)u_i, v_i>
                auto opNonlin = makeOperator(tag::test_gradtrial{},
                                             density * this->problem().solution(_v), 5);
                this->problem().addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
            }

            ///extern force (gravitational force)
            auto extForce = [](auto const& x) { return FieldVector<double,  WORLDDIM>{0.0, - 0.98};};
            auto opForce = makeOperator(tag::testvec {}, density*extForce, 5);
            this->problem().addVectorOperator(opForce, _v);

        }

        ///coupling term
        auto opCoup = makeOperator(tag::testvec_trial{}, -sigma*gradPhi);
        this->problem().addMatrixOperator(opCoup, _v, _mu);
        /*
            for (int i=0; i<WORLDDIM; i++){
                auto partphi = partialDerivativeOf(phi,i);
                problem().addMatrixOperator(zot(-sigma*partphi), makeTreePath(_1,_0, i), _mu);
            }
        */
        return;
    }

    template <class Grid>
    void CHNSProblem<Grid>::fillOperators(AdaptInfo& adaptInfo) {
                            fillCahnHilliard();

                            fillNavierStokes();

                            return;
                        }

    template <class Grid>
    void CHNSProblem<Grid>::fillBoundaryConditions(AdaptInfo& adaptInfo) {

        /// introduce index short cuts
        auto _0 = Dune::Indices::_0;
        auto _1 = Dune::Indices::_1;
        auto _2 = Dune::Indices::_2;

        /// define path shortcuts
        auto _phi =makeTreePath(_0,0);
        auto _mu =makeTreePath(_0,1);
        auto _v =makeTreePath(_1,_0);
        auto _p =makeTreePath(_1,_1);
        auto _c =makeTreePath(_2);

        this->problem().addDirichletBC([](auto const& x) {return (x[1]<1.e-8 || x[1]>2-1.e-8);}, _v, _v, FieldVector<double, WORLDDIM>{0., 0.});
        this->problem().addDirichletBC([](auto const& x) {return (x[0]<1.e-8 || x[0]>1-1.e-8);}, makeTreePath(_1,_0,0), makeTreePath(_1,_0,0), 0.);
        return;
                        };

    template <class Grid>
    void CHNSProblem<Grid>::initData(AdaptInfo& adaptInfo) {
        /// introduce index short cuts
        auto _0 = Dune::Indices::_0;
        auto _1 = Dune::Indices::_1;

        auto _phi =makeTreePath(_0,0);

        /// FUNCTIONS from the previous time step
        auto phi = this->problem().solution(_phi);

                            //Initial Value and Boundary
                            int ref_int  = Parameters::get<int>("refinement->interface").value_or(10);
                            int ref_bulk = Parameters::get<int>("refinement->bulk").value_or(2);
                            GridFunctionMarker marker("interface", this->problem.grid(),
                                                      invokeAtQP([ref_int, ref_bulk](double phi) {
                                                          return phi > 0.05 && phi < 0.95 ? ref_int : ref_bulk;
                                                          }, phi));
                            this->problem().addMarker(marker);


                            //set initial value for Benchmark
                            double eps = 0.02;
                            double radius = Parameters::get<double>("parameters->radius").value_or(0.5);

                            double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
                            double centery = Parameters::get<double>("parameters->centery").value_or(0.);

                            for (int i = 0; i < 10; ++i) {
                                phi << [eps, radius, centerx, centery](auto const& x) {
                                    using Math::sqr;
                                    return 0.5*(1 - std::tanh((std::sqrt(sqr(x[0]-centerx) + sqr(x[1]-centery))- radius)/(std::sqrt(2.0)*eps)));
                                };
                                this->problem().markElements(adaptInfo);
                                this->problem().adaptGrid(adaptInfo);
                            }
                            return;
                        };

};
