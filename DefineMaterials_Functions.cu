#include "DefineMaterials_Functions.cuh"
#include "Host_Globals.cuh"
void InitialiseMaterialsArray()
{
	for (int i = 0; i < MAXMATERIALNUM; i++)
	{
		ArrayOfMaterials_h[i].Ms = 0.0;
		ArrayOfMaterials_h[i].Aex = 0.0;
		ArrayOfMaterials_h[i].damping = 0.0;
		ArrayOfMaterials_h[i].Ku = 0.0;
		ArrayOfMaterials_h[i].Bext.X[0] = 0.0;
		ArrayOfMaterials_h[i].Bext.X[1] = 0.0;
		ArrayOfMaterials_h[i].Bext.X[2] = 0.0;
		ArrayOfMaterials_h[i].ExcitationType =0;
	}
}
void WriteMaterialsArrayToDevice()
{
checkCudaErrors(cudaMemcpyToSymbol(ArrayOfMaterials, &(ArrayOfMaterials_h[0]), MAXMATERIALNUM * sizeof(MaterialProperties)));
}
MaterialHandle DefineMaterial(double Ms,double Aex,double damping,double Ku,Vector Bext,int ExtType)
{
	int N = NumberOfMaterials;

	if (NumberOfMaterials < MAXMATERIALNUM)
	{
		NumberOfMaterials++;
	}
	else{}
	
	ArrayOfMaterials_h[N-1].Ms = Ms*(1e-3);
	ArrayOfMaterials_h[N-1].Aex = Aex* (1e18);
	ArrayOfMaterials_h[N-1].damping = damping;
	ArrayOfMaterials_h[N-1].Ku = Ku;
	ArrayOfMaterials_h[N-1].Bext.X[0] = Bext.X[0]/mu;
	ArrayOfMaterials_h[N-1].Bext.X[1] = Bext.X[1]/mu;
	ArrayOfMaterials_h[N-1].Bext.X[2] = Bext.X[2]/mu;
	ArrayOfMaterials_h[N - 1].ExcitationType =abs(ExtType);
	
	return N;
}
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, double Ku, Vector Bext)
{
	int N = NumberOfMaterials;

	if (NumberOfMaterials < MAXMATERIALNUM)
	{
		NumberOfMaterials++;
	}
	else {}

	ArrayOfMaterials_h[N - 1].Ms = Ms * (1e-3);
	ArrayOfMaterials_h[N - 1].Aex = Aex * (1e18);
	ArrayOfMaterials_h[N - 1].damping = damping;
	ArrayOfMaterials_h[N - 1].Ku = Ku;
	ArrayOfMaterials_h[N - 1].Bext.X[0] = Bext.X[0] / mu;
	ArrayOfMaterials_h[N - 1].Bext.X[1] = Bext.X[1] / mu;
	ArrayOfMaterials_h[N - 1].Bext.X[2] = Bext.X[2] / mu;
	ArrayOfMaterials_h[N - 1].ExcitationType = NoExcitation;

	return N;
}
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, double Ku)
{
	int N = NumberOfMaterials;

	if (NumberOfMaterials < MAXMATERIALNUM)
	{
		NumberOfMaterials++;
	}
	else {}

	ArrayOfMaterials_h[N - 1].Ms = Ms * (1e-3);
	ArrayOfMaterials_h[N - 1].Aex = Aex * (1e18);
	ArrayOfMaterials_h[N - 1].damping = damping;
	ArrayOfMaterials_h[N - 1].Ku = Ku;
	ArrayOfMaterials_h[N - 1].Bext.X[0] = 0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[1] = 0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[2] = 0.0;
	ArrayOfMaterials_h[N - 1].ExcitationType = NoExcitation;

	return N;
}
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, Vector Bext)
{
	int N = NumberOfMaterials;

	if (NumberOfMaterials < MAXMATERIALNUM)
	{
		NumberOfMaterials++;
	}
	else {}

	ArrayOfMaterials_h[N - 1].Ms = Ms * (1e-3);
	ArrayOfMaterials_h[N - 1].Aex = Aex * (1e18);
	ArrayOfMaterials_h[N - 1].damping = damping;
	ArrayOfMaterials_h[N - 1].Ku =0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[0] = Bext.X[0] / mu;
	ArrayOfMaterials_h[N - 1].Bext.X[1] = Bext.X[1] / mu;
	ArrayOfMaterials_h[N - 1].Bext.X[2] = Bext.X[2] / mu;
	ArrayOfMaterials_h[N - 1].ExcitationType = NoExcitation;

	return N;
}
MaterialHandle DefineMaterial(double Ms, double Aex, double damping)
{
	int N = NumberOfMaterials;

	if (NumberOfMaterials < MAXMATERIALNUM)
	{
		NumberOfMaterials++;
	}
	else {}

	ArrayOfMaterials_h[N - 1].Ms = Ms * (1e-3);
	ArrayOfMaterials_h[N - 1].Aex = Aex * (1e18);
	ArrayOfMaterials_h[N - 1].damping = damping;
	ArrayOfMaterials_h[N - 1].Ku = 0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[0] = 0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[1] =0.0;
	ArrayOfMaterials_h[N - 1].Bext.X[2] = 0.0;
	ArrayOfMaterials_h[N - 1].ExcitationType = NoExcitation;

	return N;
}
void ApplyMaterialParameters(MaterialHandle handle,double Ms, double Aex, double damping, double Ku, Vector Bext,int ExtType)
{
	if(handle > MAXMATERIALNUM)
	{
		return;
	}
	ArrayOfMaterials_h[handle-1].Ms = Ms*(1e-3);
	ArrayOfMaterials_h[handle-1].Aex = Aex*(1e18);
	ArrayOfMaterials_h[handle-1].damping = damping;
	ArrayOfMaterials_h[handle-1].Ku = Ku;
	ArrayOfMaterials_h[handle-1].Bext.X[0] = Bext.X[0]/mu;
	ArrayOfMaterials_h[handle-1].Bext.X[1] = Bext.X[1]/mu;
	ArrayOfMaterials_h[handle-1].Bext.X[2] = Bext.X[2]/mu;
	ArrayOfMaterials_h[handle - 1].ExcitationType =abs(ExtType);

}