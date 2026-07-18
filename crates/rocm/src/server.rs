#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod native {
    use crate::RocmClientError;
    use std::{
        os::unix::process::CommandExt,
        path::{Path, PathBuf},
        process::Stdio,
    };
    use tokio::{
        io::AsyncWriteExt,
        process::{Child, Command},
    };

    /// Install a systemd unit for the given ROCm device id, and try to start it.
    ///
    /// Note: This method may cause race conditions, it should be called in a critical section.
    pub(crate) async fn start_server(rocm_id: u32) -> Result<Child, RocmClientError> {
        const PATH: &str = ".sp1/bin/sp1-rocm-server";

        // Get the path to where the server binary is located.
        let path = std::env::var_os("SP1_ROCM_SERVER").map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").expect("$HOME is not set")).join(PATH)
        });

        // Download the server binary if it doesn't exist.
        maybe_download_server(&path).await?;

        // Start the server binary.
        let child = start_binary(rocm_id, &path).await?;

        Ok(child)
    }

    /// Start the server binary, ideally with systemd-run. If systemd (--user) is not available,
    /// we will run the binary as a daemon.
    async fn start_binary(rocm_id: u32, path: &Path) -> Result<Child, RocmClientError> {
        let mut cmd = Command::new(path);
        if std::env::var_os("HIP_VISIBLE_DEVICES").is_none() {
            cmd.env("ROCR_VISIBLE_DEVICES", rocm_id.to_string());
        }

        cmd.kill_on_drop(true)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| RocmClientError::new_connect(e, "Could not start `sp1-rocm-server`"))
    }

    // If the server binary is not found in the path, or if it the version is not compatible,
    // download the server binary from the release page.
    async fn maybe_download_server(path: &Path) -> Result<(), RocmClientError> {
        // Check if the server binary is in the path.
        let mut download = false;
        if !path.exists() {
            // If the path doesnt exist then there shouldnt be any instances of the server running.
            download = true;
        } else {
            let version = Command::new(path).arg("--version").output().await.map_err(|e| {
                RocmClientError::new_download_io(e, "Could not check `sp1-rocm-server` version")
            })?;

            let version = String::from_utf8_lossy(&version.stdout);
            tracing::debug!("sp1-rocm-server version: {}", version);

            // If the version is not compatible, stop all instances of the server
            // and download the new version.
            if version.trim() != sp1_primitives::SP1_CRATE_VERSION {
                download = true;

                // Stop *ALL* services, so we can replace it with a new version.
                //
                // NOTE: If a user is running a ROCm prover, across different versions,
                // on the same machine, this will cause other instances to crash!
                let mut cmd = Command::new("systemctl");
                cmd.arg("--user").arg("stop").arg(r#"sp1-rocm-server-\*"#);

                let _ = cmd.status().await.map_err(|e| {
                    RocmClientError::new_download_io(e, "Could not stop `sp1-rocm-server`")
                })?;
            }
        }

        if download {
            tracing::debug!("Downloading `sp1-rocm-server`");

            let version = format!("v{}", sp1_primitives::SP1_CRATE_VERSION);
            let repo = "succinctlabs/sp1";
            let static_url = format!("https://github.com/{repo}/releases/download");
            let asset_name = format!("sp1_rocm_server_{version}_x86_64.tar.gz");

            // Create the tar file were going to extract from.
            let tar_file = path.with_extension("tar.gz");

            // Ensure that the `.sp1` directory exists.
            tokio::fs::create_dir_all(path.parent().expect("path has no parent")).await.map_err(
                |e| RocmClientError::new_download_io(e, "Could not create `.sp1` directory"),
            )?;

            let mut file = tokio::fs::File::create(&tar_file).await.map_err(|e| {
                RocmClientError::new_download_io(e, "Could not create `sp1-rocm-server` tar file")
            })?;

            // Download the release, use a token if it exists for private releases.
            let bytes = match std::env::var("DEV_GITHUB_TOKEN").ok() {
                Some(token) => download_with_auth(&version, repo, &token, &asset_name).await,
                None => {
                    // Create the static url of the release that we expect to exist.
                    let url = format!("{static_url}/{version}/{asset_name}");

                    // Download the release.
                    let client = reqwest::Client::new();
                    let response =
                        client.get(url).send().await.map_err(RocmClientError::Download)?;

                    if !response.status().is_success() {
                        return Err(RocmClientError::Unexpected(format!(
                            "Failed to download ROCm server: {}",
                            response.text().await.expect("failed to read response text")
                        )));
                    }

                    response.bytes().await.map_err(RocmClientError::Download)
                }
            }?;

            file.write_all(&bytes).await.map_err(|e| {
                RocmClientError::new_download_io(e, "Could not write `sp1-rocm-server` tar file")
            })?;

            // Extract the tar file.
            let mut cmd = Command::new("tar");
            cmd.arg("-xzf")
                .arg(&tar_file)
                .arg("-C")
                .arg(path.parent().expect("path has no parent"));

            cmd.status().await.map_err(|e| {
                RocmClientError::new_download_io(e, "Could not extract `sp1-rocm-server` tar file")
            })?;

            // Remove the tar file.
            tokio::fs::remove_file(tar_file).await.map_err(|e| {
                RocmClientError::new_download_io(e, "Could not remove `sp1-rocm-server` tar file")
            })?;
        }

        Ok(())
    }

    async fn download_with_auth(
        tag: &str,
        repo: &str,
        token: &str,
        asset_name: &str,
    ) -> Result<bytes::Bytes, RocmClientError> {
        tracing::trace!("downloading with auth");

        // 1. Find the release by tag
        #[derive(serde::Deserialize)]
        struct Release {
            assets: Vec<Asset>,
        }
        #[derive(serde::Deserialize)]
        struct Asset {
            id: u64,
            name: String,
        }

        let api = format!("https://api.github.com/repos/{repo}");
        let client = reqwest::Client::builder()
            .user_agent("sp1-rocm-downloader")
            .build()
            .expect("failed to build reqwest client");

        let release: Release = client
            .get(format!("{api}/releases/tags/{tag}"))
            .bearer_auth(token)
            .send()
            .await
            .map_err(RocmClientError::Download)?
            .error_for_status()
            .map_err(RocmClientError::Download)?
            .json()
            .await
            .map_err(RocmClientError::Download)?;

        let asset_id = release
            .assets
            .into_iter()
            .find(|a| a.name == asset_name)
            .ok_or_else(|| {
                RocmClientError::Unexpected(format!(
                    "asset {asset_name} not found in release {tag}"
                ))
            })?
            .id;

        // 2. Download the binary content
        let bytes = client
            .get(format!("{api}/releases/assets/{asset_id}"))
            .bearer_auth(token)
            .header(reqwest::header::ACCEPT, "application/octet-stream")
            .send()
            .await
            .map_err(RocmClientError::Download)?
            .error_for_status()
            .map_err(RocmClientError::Download)?
            .bytes()
            .await
            .map_err(RocmClientError::Download)?;

        Ok(bytes)
    }

    /// The name of the systemd unit for the given ROCm device id.
    fn unit_name(rocm_id: u32) -> String {
        format!("sp1-rocm-server-{rocm_id}")
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod native {
    use crate::RocmClientError;
    use tokio::process::Child;

    pub(crate) async fn start_server(rocm_id: u32) -> Result<Child, RocmClientError> {
        panic!("Unsupported architecture for ROCm server: {}", std::env::consts::ARCH);
    }
}

pub(crate) use native::start_server;
