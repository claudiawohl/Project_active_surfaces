//
// Created by claudia on 29.09.2021.
//
#include <amdis/AMDiS.hpp>
#include <amdis/GridFunctions.hpp>

#include <amdis/AdaptInstationary.hpp>
#include "amdis/extensions/BaseProblem.hpp"

using namespace AMDiS;

template <class Grid>
using concTraits = LagrangeBasis<Grid, 2>;

template <class Grid, class Phase>
        struct ConcProb: BaseProblem<concTraits<Grid>>

            {
            using Super = BaseProblem<concTraits<Grid>>;
            Phase phi_;

            ConcProb(std::string name, Grid& grid, Phase const& phi)
            : Super(name, grid),
            phi_(phi) {}

            public:

                int _c= Dune::Indices::_0;

                bool axisymmetric = Parameters::get<bool>("parameters->axisymmetric").value_or(false);
                double m = Parameters::get<double>("parameters->mobility").value_or(0.4);
                double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*sqrt(2.)*3.;
                double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
                double extra = Parameters::get<double>("parameters->extra").value_or(1e-4);
                double invtau = 1./Parameters::get<double>("adapt->timestep").value_or(1);

                auto gradPhi() const     {return gradientOf(evalAtQP(phi_));}
                auto absGradPhi()        {return max(two_norm(gradPhi()), extra);}
                auto normal_vec() {return gradPhi()/absGradPhi();}
                auto Id() {return 1.;}
                auto NxN() {return outer(normal_vec(),normal_vec());}

                //double radius = Parameters::get<double>("parameters->radius").value_or(0.5);
                //double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
                //double centery = Parameters::get<double>("parameters->centery").value_or(0.);

            public:

                void fillOperators(AdaptInfo& adaptInfo) override {
                    auto c = this->solution();
                    auto gradC = gradientOf(c);

                    //time derivative
                    this->problem().addMatrixOperator(zot(invtau*absGradPhi(), 15));
                    this->problem().addVectorOperator(zot(absGradPhi()*invtau*c, 15));

                    //Laplace term
                    auto opConcLaplacian1 = makeOperator(tag::gradtest_gradtrial {}, absGradPhi() , 15);
                    this->problem().addMatrixOperator(opConcLaplacian1);
                    //Maybe change sign here...
                    //auto opConcLaplacian2 = makeOperator(tag::gradtest_gradtrial {}, - absGradPhi * NxN(),5);
                    //this->problem().addMatrixOperator(opConcLaplacian2, _c, _c);

                    if(axisymmetric){
                        //auto PGradC_r = absGradPhi()*Dune::FieldVector<double, 2>{1., 0.}*gradC;
                        //auto PGradC_r = (FieldVector<double, 2>{1.,0.}-FieldVector<double, 2>{1.,0.}*NxN())*gradC;
                        //this->problem().addVectorOperator(zot(PGradC_r/X(0), 5));

                        auto PGradC_r = makeOperator(tag::test_gradtrial {}, -absGradPhi()*FieldVector<double, 2>{1.,0.}, 5);
                        //auto PGradC_r = makeOperator(tag::test_gradtrial {}, absGradPhi()*(FieldVector<double, 2>{1.,0.}-FieldVector<double, 2>{1.,0.}*NxN()), 5);
                        this->problem().addMatrixOperator(PGradC_r);

                    }

                }

                void fillBoundaryConditions(AdaptInfo& adaptInfo) override {
                    return;
                }

                void solveInitialProblem(AdaptInfo& adaptInfo) override {
                    auto c = this->problem().solution();
                    int i = 0;
                    if (axisymmetric){i = 1;}

                    c << [i](auto const& x){
                        if (x[i]<0.5){return 1.5;};
                        return 1.0;
                    };
                    return;
                }

        };
