/*
 *  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     *  Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     *  Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     *  Neither the name of the NVIDIA CORPORATION nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CommandLine.hpp"
#include "Solver.hpp"
#include <stdio.h>

using namespace poisson;

//------------------------------------------------------------------------

void printUsage(void)
{
    printf("\n");
    printf("Usage: poisson.exe [OPTIONS]\n");
    printf("Reconstruct an image from its gradients by solving the screened Poisson equation.\n");
    printf("Tero Karras (tkarras@nvidia.com)\n");
    printf("\n");
    printf("Input images in PFM format:\n");
    printf("  -dx         <PFM>  Noisy horizontal gradient image. Typically '<BASE>-dx.pfm'.\n");
    printf("  -dy         <PFM>  Noisy vertical gradient image. Default is '<BASE>-dy.pfm' based on -dx.\n");
    printf("  -throughput <PFM>  Noisy throughput image. Default is '<BASE>-throughput.pfm' based on -dx.\n");
    printf("  -direct     <PFM>  Direct light image. Default is '<BASE>-direct.pfm' based on -dx.\n");
    printf("  -reference  <PFM>  Reference image for PSNR. Default is '<BASE>-reference.pfm' based on -dx.\n");
    printf("  -alpha      <0.2>  How much weight to put on the throughput image compared to the gradients.\n");
    printf("\n");
    printf("Output images in PFM format:\n");
    printf("  -indirect   <PFM>  Solved indirect light image. Default is '<BASE>-indirect.pfm' based on -dx.\n");
    printf("  -final      <PFM>  Direct plus indirect. Default is '<BASE>-final.pfm' based on -dx.\n");
    printf("  -noindirect        Do not output indirect light image.\n");
    printf("  -nofinal           Do not output final image.\n");
    printf("  -nopfm             Do not output any PFM images.\n");
    printf("\n");
    printf("PNG conversion:\n");
    printf("  -brightness 1.0    Scale image intensity before converting to sRGB color space.\n");
    printf("  -pngin             Convert all input images to PNG. By default, this is done for PNG files that do not exist.\n");
    printf("  -nopngin           Do not convert input images to PNG.\n");
    printf("  -pngout            Convert all output images to PNG. This is the default.\n");
    printf("  -nopngout          Do not convert output images to PNG.\n");
    printf("  -nopng             Do not output any PNG images.\n");
    printf("\n");
    printf("Other options:\n");
    printf("  -backend  CUDA     Enable GPU acceleration using CUDA. Requires a GPU with compute capability 3.0 or higher.\n");
    printf("  -backend  OpenMP   Enable multicore acceleration using OpenMP.\n");
    printf("  -backend  Naive    Use naive single-threaded CPU implementation.\n");
    printf("  -backend  Auto     Use 'CUDA' if available, or fall back to 'OpenMP' if not. This is the default.\n");
    printf("  -device   0        Choose the CUDA device to use. Only applicable to 'CUDA' and 'Auto'.\n");
    printf("  -verbose, -v       Enable verbose printouts.\n");
    printf("  -display, -d       Display progressive image refinement during the solver.\n");
    printf("  -help,    -h       Display this help text.\n");
    printf("\n");
    printf("Solver presets (default is L1D):\n");
    printf("  -config  L1D      L1 default config: ~1s for 1280x720 on GTX980, L1 error lower than MATLAB reference.\n");
    printf("  -config  L1Q      L1 high-quality config: ~50s for 1280x720 on GTX980, L1 error as low as possible.\n");
    printf("  -config  L1L      L1 legacy config: ~89s for 1280x720 on GTX980, L1 error equal to MATLAB reference.\n");
    printf("  -config  L2D      L2 default config: ~0.1s for 1280x720 on GTX980, L2 error equal to MATLAB reference.\n");
    printf("  -config  L2Q      L2 high-quality config: ~0.5s for 1280x720 on GTX980, L2 error as low as possible.\n");
    printf("\n");
    printf("Solver configuration:\n");
    printf("  -irlsIterMax 20   Number of iteratively reweighted least squares (IRLS) iterations.\n");
    printf("  -irlsRegInit 0.05 Initial value of the IRLS regularization parameter.\n");
    printf("  -irlsRegIter 0.5  Multiplier for the IRLS regularization parameter on subsequent iterations.\n");
    printf("  -cgIterMax   50   Maximum number of conjugate gradient (CG) iterations per IRLS iteration.\n");
    printf("  -cgIterCheck 100  Check status every N iterations (incl. early exit, CPU-GPU sync, printouts, image display).\n");
    printf("  -cgPrecond   0    0 = regular conjugate gradient (optimized), 1 = preconditioned conjugate gradient (experimental).\n");
    printf("  -cgTolerance 0    Stop CG iteration when the weight L2 error  (errL2\n");
    printf("\n");
    printf("Example:\n");
    printf("  poisson.exe -dx scenes/bathroom-dx.pfm -alpha 0.2 -brightness 2\n");
}

//------------------------------------------------------------------------

bool fileExists(const std::string& path)
{
    FILE* f = NULL;
    fopen_s(&f, path.c_str(), "rb");
    if (f)
        fclose(f);
    return (f != NULL);
}

//------------------------------------------------------------------------

void fixInputImagePaths(std::string& pfmPath, std::string& pngPath, const char* suffix, const std::string& basePath, bool pngIn, bool noPngIn)
{
    // No PFM path specified => generate one from base path.

    if (!pfmPath.length() && basePath.length())
    {
        std::string path = basePath + suffix + ".pfm";
        if (fileExists(path))
            pfmPath = path;
    }

    // No PNG path specified => generate one from PFM path.

    if (pfmPath.length() && !pngPath.length())
        pngPath = pfmPath.substr(0, pfmPath.rfind('.')) + ".png";

    // Do we want to create the PNG?

    if (!pfmPath.length() || noPngIn || (!pngIn && pngPath.length() && fileExists(pngPath)))
        pngPath = "";
}

//------------------------------------------------------------------------

void fixOutputImagePaths(std::string& pfmPath, std::string& pngPath, const char* suffix, const std::string& basePath, bool noOut, bool pngOut)
{
    // Disabled => do not output PFM.
    // PFM path not specified => generate one from base path.

    if (noOut)
        pfmPath = "";
    else if (!pfmPath.length() && basePath.length())
        pfmPath = basePath + suffix + ".pfm";

    // Disabled => do not output PNG.
    // PNG path not specified => generate one from PFM path.

    if (!pfmPath.length() || !pngOut)
        pngPath = "";
    else if (!pngPath.length())
        pngPath = pfmPath.substr(0, pfmPath.rfind('.')) + ".png";
}

//------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Parse command line.

    Solver::Params p;
    bool        alphaGiven  = false;
    bool        noIndirect  = false;
    bool        noFinal     = false;
    bool        pngIn       = false;
    bool        noPngIn     = false;
    bool        pngOut      = true;
    bool        help        = false;
    std::string error       = "";

    for (int i = 1; i < argc; i++)
    {
             if (strcmp(argv[i], "-dx")         == 0 && i + 1 < argc)   p.dxPFM         = argv[++i];
        else if (strcmp(argv[i], "-dy")         == 0 && i + 1 < argc)   p.dyPFM         = argv[++i];
        else if (strcmp(argv[i], "-throughput") == 0 && i + 1 < argc)   p.throughputPFM = argv[++i];
        else if (strcmp(argv[i], "-direct")     == 0 && i + 1 < argc)   p.directPFM     = argv[++i];
        else if (strcmp(argv[i], "-reference")  == 0 && i + 1 < argc)   p.referencePFM  = argv[++i];
        else if (strcmp(argv[i], "-alpha")      == 0 && i + 1 < argc)   p.alpha         = std::stof(argv[++i]), alphaGiven = true;
        else if (strcmp(argv[i], "-indirect")   == 0 && i + 1 < argc)   p.indirectPFM   = argv[++i];
        else if (strcmp(argv[i], "-final")      == 0 && i + 1 < argc)   p.finalPFM      = argv[++i];
        else if (strcmp(argv[i], "-noindirect") == 0)                   noIndirect      = true;
        else if (strcmp(argv[i], "-nofinal")    == 0)                   noFinal         = true;
        else if (strcmp(argv[i], "-nopfm")      == 0)                   noIndirect      = true, noFinal = true;
        else if (strcmp(argv[i], "-brightness") == 0 && i + 1 < argc)   p.brightness    = std::stof(argv[++i]);
        else if (strcmp(argv[i], "-pngin")      == 0)                   pngIn           = true, noPngIn = false;
        else if (strcmp(argv[i], "-nopngin")    == 0)                   noPngIn         = true, pngIn = false;
        else if (strcmp(argv[i], "-pngout")     == 0)                   pngOut          = true;
        else if (strcmp(argv[i], "-nopngout")   == 0)                   pngOut          = false;
        else if (strcmp(argv[i], "-nopng")      == 0)                   pngIn           = false, noPngIn = true, pngOut = false;
        else if (strcmp(argv[i], "-backend")    == 0 && i + 1 < argc)   p.backend       = argv[++i];
        else if (strcmp(argv[i], "-device")     == 0 && i + 1 < argc)   p.cudaDevice    = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-verbose")    == 0)                   p.verbose       = true;
        else if (strcmp(argv[i], "-v")          == 0)                   p.verbose       = true;
        else if (strcmp(argv[i], "-display")    == 0)                   p.display       = true;
        else if (strcmp(argv[i], "-d")          == 0)                   p.display       = true;
        else if (strcmp(argv[i], "-help")       == 0)                   help            = true;
        else if (strcmp(argv[i], "-h")          == 0)                   help            = true;
        else if (strcmp(argv[i], "-config")     == 0 && i + 1 < argc)
        {
            const char* preset = argv[++i];
            if (!p.setConfigPreset(preset))
                if (!error.length())
                    error = poisson::sprintf("Invalid config preset '%s'!", preset);
        }
        else if (strcmp(argv[i], "-irlsIterMax")    == 0 && i + 1 < argc)   p.irlsIterMax   = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-irlsRegInit")    == 0 && i + 1 < argc)   p.irlsRegInit   = std::stof(argv[++i]);
        else if (strcmp(argv[i], "-irlsRegIter")    == 0 && i + 1 < argc)   p.irlsRegIter   = std::stof(argv[++i]);
        else if (strcmp(argv[i], "-cgIterMax")      == 0 && i + 1 < argc)   p.cgIterMax     = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-cgIterCheck")    == 0 && i + 1 < argc)   p.cgIterCheck   = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-cgPrecond")      == 0 && i + 1 < argc)   p.cgPrecond     = (std::stoi(argv[++i]) != 0);
        else if (strcmp(argv[i], "-cgTolerance")    == 0 && i + 1 < argc)   p.cgTolerance   = std::stof(argv[++i]);
        else
        {
            if (!error.length())
                error = poisson::sprintf("Invalid command line option '%s'!", argv[i]);
        }
    }

    // Extract the base path from -dx.

    std::string basePath = p.dxPFM;
    basePath = basePath.substr(0, max((int)basePath.length() - (int)strlen("-dx.pfm"), 0));
    if (p.dxPFM != basePath + "-dx.pfm")
        basePath = "";

    // Fix image paths.

    fixInputImagePaths  (p.dxPFM,           p.dxPNG,            "-dx",          basePath, pngIn,        noPngIn);
    fixInputImagePaths  (p.dyPFM,           p.dyPNG,            "-dy",          basePath, pngIn,        noPngIn);
    fixInputImagePaths  (p.throughputPFM,   p.throughputPNG,    "-throughput",  basePath, pngIn,        noPngIn);
    fixInputImagePaths  (p.directPFM,       p.directPNG,        "-direct",      basePath, pngIn,        noPngIn);
    fixInputImagePaths  (p.referencePFM,    p.referencePNG,     "-reference",   basePath, pngIn,        noPngIn);
    fixOutputImagePaths (p.indirectPFM,     p.indirectPNG,      "-indirect",    basePath, noIndirect,   pngOut);
    fixOutputImagePaths (p.finalPFM,        p.finalPNG,         "-final",       basePath, noFinal,      pngOut);

    // Check for required parameters.

    if (!p.dxPFM.length() && !error.length())
        error = "Horizontal gradient image (-dx) not specified!";

    if (!p.dyPFM.length() && !error.length())
        error = "Cannot deduce -dy from -dx!";

    if (p.throughputPFM.length() != 0 && !alphaGiven && !error.length())
        error = "Throughput image weight (-alpha) not specified!";

    // Error => print usage.

    if (error.length() || help)
    {
        printUsage();
        if (error.length() && !help)
        {
            printf("\nError: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }

    // Run solver.

    Solver solver(p);
    if (p.verbose)
        printf("\nImport images:\n");

    solver.importImages();

    if (p.verbose)
        printf("\nSetup backend:\n");

    solver.setupBackend();

    if (p.verbose)
        printf("\nSolve indirect:\n");

    solver.solveIndirect();

    if (p.verbose)
        printf("\nEvaluate metrics:\n");
    solver.evaluateMetrics();

    if (p.verbose)
        printf("\nExport images:\n");

    solver.exportImages();

    if (p.verbose)
        printf("\nDone.\n\n");
    return 0;
}

//------------------------------------------------------------------------
