FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu20.04


ENV LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    ND_ENTRYPOINT="/neurodocker/startup.sh"

RUN export ND_ENTRYPOINT="/neurodocker/startup.sh" \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           unzip \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG="en_US.UTF-8" \
    && chmod 777 /opt && chmod a+s /opt \
    && mkdir -p /neurodocker \
    && if [ ! -f "$ND_ENTRYPOINT" ]; then \
         echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT" \
    &&   echo 'set -e' >> "$ND_ENTRYPOINT" \
    &&   echo 'export USER="${USER:=`whoami`}"' >> "$ND_ENTRYPOINT" \
    &&   echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT"; \
    fi \
    && chmod -R 777 /neurodocker && chmod a+s /neurodocker

ENV OS="Linux" \
    PATH="/opt/freesurfer-7.4.1/bin:/opt/freesurfer-7.4.1/fsfast/bin:/opt/freesurfer-7.4.1/tktools:/opt/freesurfer-7.4.1/mni/bin:$PATH" \
    FREESURFER_HOME="/opt/freesurfer-7.4.1" \
    FREESURFER="/opt/freesurfer-7.4.1" \
    SUBJECTS_DIR="/opt/freesurfer-7.4.1/subjects" \
    LOCAL_DIR="/opt/freesurfer-7.4.1/local" \
    FSFAST_HOME="/opt/freesurfer-7.4.1/fsfast" \
    FMRI_ANALYSIS_DIR="/opt/freesurfer-7.4.1/fsfast" \
    FUNCTIONALS_DIR="/opt/freesurfer-7.4.1/sessions" \
    FS_OVERRIDE="0" \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    MINC_BIN_DIR="/opt/freesurfer-7.4.1/mni/bin" \
    MINC_LIB_DIR="/opt/freesurfer-7.4.1/mni/lib" \
    MNI_DIR="/opt/freesurfer-7.4.1/mni" \
    MNI_DATAPATH="/opt/freesurfer-7.4.1/mni/data" \
    MNI_PERL5LIB="/opt/freesurfer-7.4.1/mni/share/perl5" \
    PERL5LIB="/opt/freesurfer-7.4.1/mni/share/perl5"

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           ca-certificates \
           curl \
           libgomp1 \
           libxmu6 \
           libxt6 \
           perl \
           tcsh \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FreeSurfer ..." \
    && mkdir -p /opt/freesurfer-7.4.1 \
    && curl -fL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-centos7_x86_64-7.4.1.tar.gz \
    | tar -xz -C /opt/freesurfer-7.4.1 --owner root --group root --no-same-owner --transform='s,freesurfer/,,' \
         --exclude='average/mult-comp-cor' \
         --exclude='lib/cuda' \
         --exclude='lib/qt' \
         --exclude='subjects/V1_average' \
         --exclude='subjects/bert' \
         --exclude='subjects/cvs_avg35' \
         --exclude='subjects/cvs_avg35_inMNI152' \
         --exclude='subjects/fsaverage3' \
         --exclude='subjects/fsaverage4' \
         --exclude='subjects/fsaverage5' \
         --exclude='subjects/fsaverage6' \
         --exclude='subjects/fsaverage_sym' \
         --exclude='trctrain'

ENV ANTSPATH="/opt/ants-2.4.3/" \
    PATH="/opt/ants-2.4.3:$PATH"

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ca-certificates \
           curl \
           unzip \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading ANTs ..." \
    && curl -fsSL -o ants.zip https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-centos7-X64-gcc.zip \
    && unzip ants.zip -d /opt \
    && mv /opt/ants-2.4.3/bin/* /opt/ants-2.4.3 \
    && rm ants.zip

ENTRYPOINT ["/neurodocker/startup.sh"]