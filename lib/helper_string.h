/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRCASECMP
#define STRCASECMP _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#ifndef SPRINTF
#define SPRINTF sprintf_s
#endif
#else // Linux Includes
#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#ifndef SPRINTF
#define SPRINTF sprintf
#endif
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
inline char *sdkFindFilePath(const char *filename,
                             const char *executable_path)
{
    // <executable_name> defines a variable that is replaced with the name of the
    // executable

    // Typical relative search paths to locate needed companion files (e.g. sample
    // input data, or JIT source files) The origin for the relative search may be
    // the .exe file, a .bat file launching an .exe, a browser .exe launching the
    // .exe or .bat, etc
    const char *searchPath[] = {
        "./",      // same dir
        "./data/", // same dir

        "../../../../Samples/<executable_name>/", // up 4 in tree
        "../../../Samples/<executable_name>/",    // up 3 in tree
        "../../Samples/<executable_name>/",       // up 2 in tree

        "../../../../Samples/<executable_name>/data/", // up 4 in tree
        "../../../Samples/<executable_name>/data/",    // up 3 in tree
        "../../Samples/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/0_Introduction/<executable_name>/", // up 4 in tree
        "../../../Samples/0_Introduction/<executable_name>/",    // up 3 in tree
        "../../Samples/0_Introduction/<executable_name>/",       // up 2 in tree

        "../../../../Samples/1_Utilities/<executable_name>/", // up 4 in tree
        "../../../Samples/1_Utilities/<executable_name>/",    // up 3 in tree
        "../../Samples/1_Utilities/<executable_name>/",       // up 2 in tree

        "../../../../Samples/2_Concepts_and_Techniques/<executable_name>/", // up 4 in tree
        "../../../Samples/2_Concepts_and_Techniques/<executable_name>/",    // up 3 in tree
        "../../Samples/2_Concepts_and_Techniques/<executable_name>/",       // up 2 in tree

        "../../../../Samples/3_CUDA_Features/<executable_name>/", // up 4 in tree
        "../../../Samples/3_CUDA_Features/<executable_name>/",    // up 3 in tree
        "../../Samples/3_CUDA_Features/<executable_name>/",       // up 2 in tree

        "../../../../Samples/4_CUDA_Libraries/<executable_name>/", // up 4 in tree
        "../../../Samples/4_CUDA_Libraries/<executable_name>/",    // up 3 in tree
        "../../Samples/4_CUDA_Libraries/<executable_name>/",       // up 2 in tree

        "../../../../Samples/5_Domain_Specific/<executable_name>/", // up 4 in tree
        "../../../Samples/5_Domain_Specific/<executable_name>/",    // up 3 in tree
        "../../Samples/5_Domain_Specific/<executable_name>/",       // up 2 in tree

        "../../../../Samples/6_Performance/<executable_name>/", // up 4 in tree
        "../../../Samples/6_Performance/<executable_name>/",    // up 3 in tree
        "../../Samples/6_Performance/<executable_name>/",       // up 2 in tree

        "../../../../Samples/0_Introduction/<executable_name>/data/", // up 4 in tree
        "../../../Samples/0_Introduction/<executable_name>/data/",    // up 3 in tree
        "../../Samples/0_Introduction/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/1_Utilities/<executable_name>/data/", // up 4 in tree
        "../../../Samples/1_Utilities/<executable_name>/data/",    // up 3 in tree
        "../../Samples/1_Utilities/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/2_Concepts_and_Techniques/<executable_name>/data/", // up 4 in tree
        "../../../Samples/2_Concepts_and_Techniques/<executable_name>/data/",    // up 3 in tree
        "../../Samples/2_Concepts_and_Techniques/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/3_CUDA_Features/<executable_name>/data/", // up 4 in tree
        "../../../Samples/3_CUDA_Features/<executable_name>/data/",    // up 3 in tree
        "../../Samples/3_CUDA_Features/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/4_CUDA_Libraries/<executable_name>/data/", // up 4 in tree
        "../../../Samples/4_CUDA_Libraries/<executable_name>/data/",    // up 3 in tree
        "../../Samples/4_CUDA_Libraries/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/5_Domain_Specific/<executable_name>/data/", // up 4 in tree
        "../../../Samples/5_Domain_Specific/<executable_name>/data/",    // up 3 in tree
        "../../Samples/5_Domain_Specific/<executable_name>/data/",       // up 2 in tree

        "../../../../Samples/6_Performance/<executable_name>/data/", // up 4 in tree
        "../../../Samples/6_Performance/<executable_name>/data/",    // up 3 in tree
        "../../Samples/6_Performance/<executable_name>/data/",       // up 2 in tree

        "../../../../Common/data/", // up 4 in tree
        "../../../Common/data/",    // up 3 in tree
        "../../Common/data/"        // up 2 in tree
    };

    // Extract the executable name
    std::string executable_name;

    if (executable_path != 0)
    {
        executable_name = std::string(executable_path);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != std::string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }

#else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0, delimiter_pos + 1);
#endif
    }

    // Loop over all search paths and return the first hit
    for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i)
    {
        std::string path(searchPath[i]);
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath
        // replace it with the value
        if (executable_name_pos != std::string::npos)
        {
            if (executable_path != 0)
            {
                path.replace(executable_name_pos, strlen("<executable_name>"),
                             executable_name);
            }
            else
            {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }

#ifdef _DEBUG
        printf("sdkFindFilePath <%s> in %s\n", filename, path.c_str());
#endif

        // Test if the file exists
        path.append(filename);
        FILE *fp;
        FOPEN(fp, path.c_str(), "rb");

        if (fp != NULL)
        {
            fclose(fp);
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char *file_path = reinterpret_cast<char *>(malloc(path.length() + 1));
            STRCPY(file_path, path.length() + 1, path.c_str());
            return file_path;
        }

        if (fp)
        {
            fclose(fp);
        }
    }

    // File not found
    printf("\nerror: sdkFindFilePath: file <%s> not found!\n", filename);
    return 0;
}

#endif // COMMON_HELPER_STRING_H_