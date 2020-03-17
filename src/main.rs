use human_resource_machine::{floor, inbox, run, OfficeState, OfficeTile};
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let inbox = inbox!('b', 'r', 'a', 'i', 'n', 0);
    let floor = floor!(len 15, {14, 0});
    let mut office_state = OfficeState::new(inbox, floor);
    let mut file = File::open("example.hrm")?;
    run(&mut file, &mut office_state)?;
    Ok(())
}
