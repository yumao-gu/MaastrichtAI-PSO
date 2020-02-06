#include <iostream>
#include <cmath>
#include "/home/lenovo/matplotlib-cpp/matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

#define PI 3.1415926
#define POPSIZE 20                         //粒子个数
#define MAXINTERATION 1000                 //最大迭代次数
#define NVARS 2                           //参数个数
#define WMAX 0.9                           //惯量权重最大值
#define WMIN 0.4                           //惯量权重最小值

struct particle {                          //单个粒子
    double pBest[NVARS];
    double v[NVARS];
    double x[NVARS];
    double upper[NVARS];
    double lower[NVARS];
};

double w;                                  //惯量权重
double c1 = 2.0;                           //加速因子1
double c2 = 2.0;                           //加速因子2
double absbound;                           //上下界绝对值
double vmax;                               //最大速度
double gBest[NVARS];                       //全局最优解
particle particles[POPSIZE];               //粒子群

double evalfunc(double[], int);            //评估函数
double avg(double[], int);                 //求平均数
double stddev(double[], int);              //求标准差
void initialize(int);                      //初始化函数
double randval(double, double);            //求范围（lower，upper）之间的随机数
void update(int, int);                     //利用更新公式更新每个粒子的速度和位置以及历史最优位置
void fit(void);                            //更新群体历史最优位置
void show();                               //可视化

double avg(double parameter[], int n)
{
    double num = 0;
    for (int i = 0; i < n; i++)
    {
        num += parameter[i];
    }
    return num / n;
}

double stddev(double parameter[], int n)
{
    double num = avg(parameter, n);
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += (parameter[i] - num)*(parameter[i] - num);
    }
    return sqrt(sum / n);
}

double randval(double low, double high)
{
    return (low + (high - low)*rand()*1.0 / RAND_MAX);
}

double evalfunc_Rosenbrock(double parameter[])
{
    double val = 0;
    val = pow(parameter[0],2) + 1 * pow((parameter[1] - pow(parameter[0],2)),2);
    return val;
}

double evalfunc_Rastrigin(double parameter[])
{
    double val = 0;
    for (int i = 0; i < NVARS; i++) {
        val += (parameter[i] * parameter[i] - 10 * cos(2 * PI*parameter[i] / 180 * PI) + 10.0);
    }
    return val;
}

double evalfunc(double parameter[])
{
    double val = 0;
    for (int i = 0; i < NVARS; i++) {
        val += (parameter[i] * parameter[i] - 10 * cos(2 * PI*parameter[i] / 180 * PI) + 10.0);
    }
    return val;
}

/*  初始化每个粒子的速度、位置并将  */
/*  该位置设定为当前历史最优位置找  */
/*  到所有粒子的最优位置并设定为当  */
/*  前群体历史最优位置              */
void initialize()
{
    int i, j;
    double lbound[NVARS];
    double ubound[NVARS];

    for (i = 0; i < NVARS; i++)
    {
        lbound[i] = (-absbound);
        ubound[i] = absbound;
    }

    for (i = 0; i < NVARS; i++)
    {
        for (j = 0; j < POPSIZE; j++)
        {
            particles[j].upper[i] = ubound[i];
            particles[j].lower[i] = lbound[i];
            particles[j].v[i] = randval(lbound[i], ubound[i]);
            particles[j].x[i] = randval(lbound[i], ubound[i]);
            particles[j].pBest[i] = particles[j].x[i];
        }
    }

    double pval = evalfunc(particles[0].pBest);
    int num;
    for (i = 1; i < POPSIZE; i++) {
        if (pval > evalfunc(particles[i].pBest))
        {
            pval = evalfunc(particles[i].pBest);
            num = i;
        }
    }
    for (j = 0; j < NVARS; j++)
    {
        gBest[j] = particles[num].pBest[j];
    }
}

/*  通过传入参数FUNC来调用不同的评  */
/*  估函数并对粒子位置进行评估并更  */
/*  新粒子历史最优位置              */
void evaluate()
{
    int i, j;
    double pval[POPSIZE], nval[POPSIZE];

    for (i = 0; i < POPSIZE; i++)
    {
        pval[i] = evalfunc(particles[i].pBest);
        nval[i] = evalfunc(particles[i].x);

        if (pval[i] > nval[i])
        {
            for (j = 0; j < NVARS; j++)
            {
                particles[i].pBest[j] = particles[i].x[j];
            }
        }
    }
}

/*  通过参数w_change_method来选择不 */
/*  同的惯性权重衡量规则来根据群体  */
/*  历史最优位置更新粒子速度以及位置*/
void update(int interation)
{
    int i, j;
    double v, x;

    w = WMAX - (WMAX - WMIN) / MAXINTERATION*(double)interation;

    for (i = 0; i < NVARS; i++)
    {
        for (j = 0; j < POPSIZE; j++)
        {
            v = w*particles[j].v[i] + c1*randval(0, 1)*(particles[j].pBest[i] - particles[j].x[i]) + c2*randval(0, 1)*(gBest[i] - particles[j].x[i]);
            if (v > vmax)
            {
                particles[j].v[i] = vmax;
            }
            else if (v < (-vmax))
            {
                particles[j].v[i] = (-vmax);
            }
            else
            {
                particles[j].v[i] = v;
            }

            x = particles[j].x[i] + particles[j].v[i] * 1;
            if (x > particles[j].upper[i])
            {
                particles[j].x[i] = particles[j].upper[i];
            }
            else if (x < particles[j].lower[i])
            {
                particles[j].x[i] = particles[j].lower[i];
            }
            else
            {
                particles[j].x[i] = x;
            }
        }
    }
}

/*  更新群体最优位置         */
void fit(void)
{
    int i, j;
    double gval = evalfunc(gBest);
    double pval[POPSIZE];

    for (i = 0; i < POPSIZE; i++)
    {
        pval[i] = evalfunc(particles[i].pBest);
        if (gval > pval[i])
        {
            for (j = 0; j < NVARS; j++)
            {
                gBest[j] = particles[i].pBest[j];
            }
        }
    }
}

void show()
{
    std::vector<std::vector<double>> x_map, y_map, z_map;
    for (double i = -2; i <= 2;  i += 0.1) {
        std::vector<double> x_row, y_row, z_row;
        for (double j = -2; j <= 2; j += 0.1) {
            x_row.push_back(i);
            y_row.push_back(j);
            //z_row.push_back(pow(i,2) + 1 * pow((j - pow(i,2)),2));
            z_row.push_back((i * i - 10 * cos(2 * PI*i / 180 * PI) + 10.0) + (j * j - 10 * cos(2 * PI*j / 180 * PI) + 10.0));
        }
        x_map.push_back(x_row);
        y_map.push_back(y_row);
        z_map.push_back(z_row);
    }

    plt::plot_surface(x_map, y_map, z_map,{{"alpha","0.0"},{"shade","False"}});
    std::vector<double> x, y, z;

    for (int j = 0; j < POPSIZE; j++)
    {
        x.push_back(particles[j].x[0]);
        y.push_back(particles[j].x[1]);
    }
    plt::scatter(x, y,20.0,{{"c","k"}});
    plt::show();
}

int main()
{
    srand(time(NULL));
    absbound = 2.0;
    vmax = absbound*0.15;

    initialize();
    evaluate();
    for (int i = 0; i < MAXINTERATION; i++)
    {
        update(i);
        evaluate();
        fit();
        if(i % 2  == 0)
        {
            show();
        }
    }
    //double evalue = evalfunc_Rosenbrock(gBest);

    std::cout << "End PSO" << std::endl;

    return 0;
}
