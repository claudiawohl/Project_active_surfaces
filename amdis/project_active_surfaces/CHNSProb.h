//
// Created by claudia on 27.09.2021.
//
#include <amdis/Marker.hpp>
#include "amdis/extensions/BaseProblem.hpp"

#include <amdis/AdaptInstationary.hpp>

using namespace AMDiS;

template <class Traits>
struct CHNSProb: BaseProblem<Traits>
    {
    template <class Grid, class BasisFactory>
    CHNSProb(std::string name, Grid& grid, const BasisFactory& bf)
    : BaseProblem<Traits>(name, grid, bf) {}

    public:

        /// introduce index short cuts
        static constexpr auto _0= Dune::Indices::_0;
        static constexpr auto _1 = Dune::Indices::_1;

        /// define path shortcuts
        Dune::TypeTree::HybridTreePath<std::integral_constant<unsigned long, 0>, unsigned long> _phi = makeTreePath(_0,0);
        Dune::TypeTree::HybridTreePath<std::integral_constant<unsigned long, 0>, unsigned long> _mu = makeTreePath(_0,1);
        Dune::TypeTree::HybridTreePath<std::integral_constant<unsigned long, 1>, std::integral_constant<unsigned long, 0> > _v = makeTreePath(_1,_0);
        Dune::TypeTree::HybridTreePath<std::integral_constant<unsigned long, 1>, std::integral_constant<unsigned long, 1> > _p = makeTreePath(_1,_1);

        bool axisymmetric = Parameters::get<bool>("parameters->axisymmetric").value_or(false);
        const double m = Parameters::get<double>("parameters->mobility").value_or(0.4);
        const double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*sqrt(2.)*3.;
        const double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);

        //double radius = Parameters::get<double>("parameters->radius").value_or(0.5);
        //double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
        //double centery = Parameters::get<double>("parameters->centery").value_or(0.);

        /// FUNCTIONS from the previous time step
        auto phi()    {return this->solution(_phi);}
        auto v() {return this->solution(_v);}
        auto phiProjected() {return max(min(evalAtQP(phi()),Dune::FieldVector<double,1>{1.}), Dune::FieldVector<double, 1>{0.});}

        //TODO: invTau, gradPhi, phi in initData(), solveInitialProblem()

    public:

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
        void fillCahnHilliard(AdaptInfo& adaptInfo) {
            auto invTau = std::ref(this->invTau());
            
            /// FIRST EQUATION
            /**\f(
              (\partial_t \phi^m, \varphi_1) + (v^{m-1}\cdot \nabla \phi^m, \varphi_1) + (M\nabla \mu^m, \nabla \varphi_1)&=0
               \f)**/


            ///time derivative
            this->problem().addMatrixOperator(zot(invTau), _phi, _phi);
            this->problem().addVectorOperator(zot(phi() * invTau), _phi);

            ///convection term (coupling)
            auto opFconv = makeOperator(tag::test_trialvec{}, gradientOf(phi()), 5);
            this->problem().addMatrixOperator(opFconv, _phi, _v);

            ///mobility term
            this->problem().addMatrixOperator(sot(m), _phi, _mu);

            if (axisymmetric)
            {
                auto opLaplaceMuAxi = makeOperator(tag::test_partialtrial{0}, -m/X(0), 3);
                this->problem().addMatrixOperator(opLaplaceMuAxi, _phi, _mu);
            };

            ///SECOND EQUATION
            /**\f(
             * (\mu^m, \varphi_2)-\sigma \varepsilon(\nabla \phi^m,\nabla \varphi_2)- \varepsilon^{-1}(W'(\phi^m),\varphi_2) &=0
             * \f)
             */

            ///mu
            this->problem().addMatrixOperator(zot(1.0), _mu, _mu);
            ///Laplace phi
            this->problem().addMatrixOperator(sot(-eps), _mu, _phi);

            if (axisymmetric)
            {
                auto opLaplacePhiAxi= makeOperator(tag::test_partialtrial{0}, eps/X(0), 3);
                this->problem().addMatrixOperator(opLaplacePhiAxi, _mu, _phi);
            };

            ///double Well potential: phi^2(1-phi)^2
            auto opFimpl = zot(-1./eps * (2. + 12*phi()*(phi() - 1.)));
            this->problem().addMatrixOperator(opFimpl, _mu, _phi);
            auto opFexpl = zot(1./eps * pow<2>(phi())*(6. - 8*phi()));
            this->problem().addVectorOperator(opFexpl, _mu);

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
        void fillNavierStokes(AdaptInfo& adaptInfo) {
            auto invTau = std::ref(this->invTau());

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

            auto viscosity = inner_visc*phiProjected()+outer_visc*(1.-phiProjected());

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
                auto density = density_inner * phiProjected() + (1. - phiProjected()) * density_outer;

                ///time derivative
                // <1/tau * u, v>
                auto opTime = makeOperator(tag::testvec_trialvec{},
                                           density * invTau, 5);
                this->problem().addMatrixOperator(opTime, _v, _v);

                // <1/tau * u^old, v>
                auto opTimeOld = makeOperator(tag::testvec{},
                                              density * invTau * v(), 5);
                this->problem().addVectorOperator(opTimeOld, _v);

                ///material derivative part
                for (int i = 0; i < WORLDDIM; ++i) {
                    // <(u^old * nabla)u_i, v_i>
                    auto opNonlin = makeOperator(tag::test_gradtrial{},
                                                 density * v(), 5);
                    this->problem().addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
                }

                ///extern force (gravitational force)
                auto extForce = [](auto const& x) { return FieldVector<double,  WORLDDIM>{0.0, - 0.98};};
                auto opForce = makeOperator(tag::testvec {}, density*extForce, 5);
                this->problem().addVectorOperator(opForce, _v);

            }

            ///coupling term
            auto opCoup = makeOperator(tag::testvec_trial{}, -sigma* gradientOf(phi()));
            this->problem().addMatrixOperator(opCoup, _v, _mu);
            /*
                for (int i=0; i<WORLDDIM; i++){
                    auto partphi = partialDerivativeOf(phi,i);
                    this->problem().addMatrixOperator(zot(-sigma*partphi), makeTreePath(_1,_0, i), _mu);
                }
            */

            return; 
        }
        
        void fillOperators(AdaptInfo& adaptInfo) override {
            this->fillCahnHilliard(adaptInfo);
            this->fillNavierStokes(adaptInfo);
        }

        void fillBoundaryConditions(AdaptInfo& adaptInfo) override {
            this->problem().addDirichletBC([](auto const& x) {return (x[1]<1.e-8 || x[1]>2-1.e-8);}, _v, _v, FieldVector<double, WORLDDIM>{0., 0.});
            this->problem().addDirichletBC([](auto const& x) {return (x[0]<1.e-8 || x[0]>1-1.e-8);}, makeTreePath(_1,_0,0), makeTreePath(_1,_0,0), 0.);
            this->problem().addDirichletBC([](auto const& x) {return (x[1]<1.e-8 );}, _p, _p, 0.);
        }

        void initData(AdaptInfo& adaptInfo) override{
            //this->oldSolution_.reset(new DOFVector<GlobalBasis>(*this->globalBasis(), DataTransferOperation::INTERPOLATE));

            /// FUNCTIONS from the previous time step
            auto phi = this->problem().solution(_phi);

            int ref_int  = Parameters::get<int>("refinement->interface").value_or(10);
            int ref_bulk = Parameters::get<int>("refinement->bulk").value_or(2);
            auto marker = GridFunctionMarker("interface", this->problem().grid(),
                                             invokeAtQP([ref_int, ref_bulk](double phi) -> int {
                                                 return phi > 0.05 && phi < 0.95 ? ref_int : ref_bulk;
                                                 }, phi));
            this->problem().addMarker(Dune::wrap_or_move(std::move(marker)));
        }

        void solveInitialProblem(AdaptInfo& adaptInfo) override{

            //TODO: Only define them as member function
            /// FUNCTIONS from the previous time step
            auto phi = this->problem().solution(_phi);

            double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
            double radius = Parameters::get<double>("parameters->radius").value_or(0.5);
            double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
            double centery = Parameters::get<double>("parameters->centery").value_or(0.);

            //set initial value for Benchmark
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

// deduction guide
template <class Grid, class BasisFactory>
        CHNSProb(std::string name, Grid& grid, const BasisFactory& bf)
        -> CHNSProb<Impl::DeducedProblemTraits_t<Grid,BasisFactory>>;
