use serde::Serialize;
use tar::{Builder, Header};

pub fn serialize_object<W: std::io::Write>(
    path: String,
    data: &impl Serialize,
    archive: &mut Builder<W>,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let data = serde_json::to_string(data)?.into_bytes();

    let mut header = Header::new_gnu();
    header.set_path(path)?;
    header.set_size(data.len() as u64);
    header.set_mode(0o664);
    header.set_cksum();

    archive.append(&header, &data as &[u8])?;
    Ok(())
}
