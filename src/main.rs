use std::error;
use std::fs::File;

use std::io::BufRead;
use std::io::BufReader;

use std::collections::HashMap;
use std::collections::VecDeque;

use std::convert::From;
use std::fmt;

/// An instruction without it argument
#[derive(Debug)]
enum Op {
    Inbox,
    Outbox,
    CopyFrom,
    CopyTo,
    Jump,
    JumpZ,
    JumpN,
    LabelDef(String),
    BumpUp,
    BumpDown,
    Sub,
    Add,
    Define(Define), //ignored, defines the shapes of drawings
}

/// The smallest executable element
#[derive(Debug, Clone)]
enum Instruction {
    Inbox,
    Outbox,
    CopyFrom(Address),
    CopyTo(Address),
    Jump(Label),
    JumpZ(Label),
    JumpN(Label),
    LabelDef(String),
    BumpUp(Address),
    BumpDown(Address),
    Sub(Address),
    Add(Address),
    Define(Define), //ignored, defines the shapes of drawings
}

#[derive(Debug, Clone)]
struct InstructionDebug(Instruction, DebugInfo);

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::Inbox => write!(f, "Inbox"),
            Instruction::Outbox => write!(f, "Outbox"),
            Instruction::CopyFrom(addr) => write!(f, "CopyFrom:{}", addr),
            Instruction::CopyTo(addr) => write!(f, "CopyTo:{}", addr),
            Instruction::Jump(label) => write!(f, "Jump:{}", label),
            Instruction::JumpZ(label) => write!(f, "JumpZ:{}", label),
            Instruction::JumpN(label) => write!(f, "JumpN:{}", label),
            Instruction::LabelDef(name) => write!(f, "LabelDef:{}", name),
            Instruction::BumpUp(addr) => write!(f, "BumpUp:{}", addr),
            Instruction::BumpDown(addr) => write!(f, "BumpDown:{}", addr),
            Instruction::Sub(addr) => write!(f, "Sub:{}", addr),
            Instruction::Add(addr) => write!(f, "Add:{}", addr),
            Instruction::Define(addr) => write!(f, "Define:{}", addr),
        }
    }
}

impl fmt::Display for InstructionDebug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
enum Define {
    Comment(usize, String),
    Label(usize, String),
}

impl fmt::Display for Define {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Define::Comment(num, comment) => write!(f, "Comment[{}]:{}", num, comment),
            Define::Label(num, label) => write!(f, "Label[{}]:{}", num, label),
        }
    }
}

#[derive(Debug, Clone)]
/// label as argument, not definition of a label
struct Label {
    name: String,
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Label:{}", self.name)
    }
}

/// Space separated syntax element
#[derive(Debug)]
enum Token {
    Op(Op),
    Address(Address),
    Label(Label),
}

/// Token that refers to an office tile on the floor
#[derive(Debug, Copy, Clone)]
enum Address {
    Address(usize),
    AddressOf(usize),
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Address::AddressOf(tile) => write!(f, "[{}]", tile),
            Address::Address(tile) => write!(f, "{}", tile),
        }
    }
}

/// A token with some extra debug info
#[derive(Debug)]
struct TokenDebug {
    token: Token,
    debug_info: DebugInfo,
}

/// The debug info of a token e.g. the line it occured in the original source
#[derive(Debug, Clone)]
struct DebugInfo {
    line: usize,
}

/// A value
/// Can either be a character or a number (integer)
/// OfficeTiles come in the inbox, are placed on the floor and go out the outbox
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OfficeTile {
    Number(i16),     //numbers in human resource machine are in -999..=999
    Character(char), //chars in human resource machine appear to be all [a-zA-Z]
}

impl From<char> for OfficeTile {
    fn from(c: char) -> OfficeTile {
        OfficeTile::Character(c)
    }
}

impl From<i16> for OfficeTile {
    fn from(n: i16) -> OfficeTile {
        OfficeTile::Number(n)
    }
}

macro_rules! create_inbox {
    ( $( $x:expr ),* ) => {
        {
            let mut inbox = VecDeque::new();
            $(
                inbox.push_back(OfficeTile::from($x));
            )*
            inbox
        }
    }
}

macro_rules! create_floor {
    ( $len:expr, $( $index:expr,  $tile:expr ),* ) => {
        {
            let mut floor = Vec::with_capacity($len);
            for _ in 0..$len {
                floor.push(None)
            }
            $(
                floor[$index] = Some(tile!($tile));
            )*
            floor
        }
    }
}

macro_rules! tile {
    ( $value:expr ) => {{
        OfficeTile::from($value)
    }};
}

impl fmt::Display for OfficeTile {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OfficeTile::Character(c) => write!(f, "{}", c),
            OfficeTile::Number(n) => write!(f, "{}", n),
        }
    }
}

impl OfficeTile {
    fn checked_add(self, rhs: OfficeTile) -> Result<OfficeTile, ArithmeticError> {
        match self {
            OfficeTile::Number(num) => match rhs {
                OfficeTile::Number(rhs_num) => match num.checked_add(rhs_num) {
                    Some(res) => {
                        if res > 999_i16 {
                            return Err(ArithmeticError::OverflowError);
                        }
                        Ok(OfficeTile::Number(res))
                    }
                    None => Err(ArithmeticError::OverflowError),
                },
                OfficeTile::Character(_) => Err(ArithmeticError::TypeError),
            },
            OfficeTile::Character(character) => match rhs {
                OfficeTile::Character(rhs_char) => {
                    match (character as i16).checked_add(rhs_char as i16) {
                        Some(res) => {
                            if res > 999_i16 {
                                return Err(ArithmeticError::OverflowError);
                            }
                            Ok(OfficeTile::Number(res))
                        }
                        None => Err(ArithmeticError::OverflowError),
                    }
                }
                OfficeTile::Number(_) => Err(ArithmeticError::TypeError),
            },
        }
    }

    fn checked_sub(self, rhs: OfficeTile) -> Result<OfficeTile, ArithmeticError> {
        match self {
            OfficeTile::Number(num) => match rhs {
                OfficeTile::Number(other_num) => {
                    let res = num - other_num;
                    if res < -999_i16 {
                        return Err(ArithmeticError::OverflowError);
                    }
                    Ok(OfficeTile::Number(res))
                }
                OfficeTile::Character(_) => Err(ArithmeticError::TypeError),
            },
            OfficeTile::Character(character) => match rhs {
                OfficeTile::Character(other_char) => {
                    let res: i16 = character as i16 - other_char as i16;
                    if res < -999_i16 {
                        return Err(ArithmeticError::OverflowError);
                    }
                    Ok(OfficeTile::Number(res))
                }
                OfficeTile::Number(_) => Err(ArithmeticError::TypeError),
            },
        }
    }
}

// TODO: add instruction pointer here
/// The state of the entire office
/// Composed of the tile held by the player, the inbox, outbox and floor
#[derive(Debug)]
struct OfficeState {
    held: Option<OfficeTile>,
    inbox: VecDeque<OfficeTile>,
    outbox: VecDeque<OfficeTile>,
    floor: Vec<Option<OfficeTile>>,
}

impl OfficeState {
    pub fn new_with_inbox_floor(
        inbox: VecDeque<OfficeTile>,
        floor: Vec<Option<OfficeTile>>,
    ) -> OfficeState {
        OfficeState {
            held: None,
            inbox,
            outbox: vec![].into_iter().collect(),
            floor,
        }
    }
}

impl fmt::Display for OfficeState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let OfficeState {
            inbox,
            outbox,
            floor,
            held,
        } = self;
        writeln!(
            f,
            "held: [{}] ",
            held.map_or(" ".to_string(), |held| held.to_string())
        )?;
        let v = vec![
            self.floor.len() / 5 + 1,
            if self.inbox.len() > 7 {
                7
            } else {
                self.inbox.len()
            },
            if self.outbox.len() > 7 {
                7
            } else {
                self.outbox.len()
            },
        ];
        let rows: usize = *v.iter().max().unwrap_or(&0);

        // Don't change this value unless you change the 11 in the format string
        let floor_width = 5;
        let floor_string_width = 2 * floor_width + 1;
        let mut s: String = String::new();
        for _ in 0..floor_string_width {
            s.push(' ')
        }
        writeln!(f, "in {} out", s)?;
        for row in 0..rows {
            let inbox_val = inbox.get(row).map_or("".to_string(), ToString::to_string);
            let outbox_val = outbox.get(row).map_or("".to_string(), ToString::to_string);
            let mut s = String::with_capacity(2 * floor_width + 1);
            s.push(' ');
            for index in row * floor_width..(row + 1) * floor_width {
                let default = " ".to_string();
                let floor_val = floor.get(index).map_or(default.clone(), |val| {
                    val.map_or(default, |val| val.to_string())
                });
                s.push_str(&floor_val[..]);
                s.push(' ');
            }
            writeln!(f, "{:^3} {:^11} {:^3}", inbox_val, s, outbox_val)?
        }
        Ok(())
    }
}

trait Addressable {
    fn get_value(&self, state: &OfficeState) -> usize;
}

impl Addressable for Address {
    fn get_value(&self, state: &OfficeState) -> usize {
        match self {
            Address::AddressOf(addr) => {
                let points_to = *state.floor.get(*addr).expect("Address out of range");
                let points_to = points_to.expect("Value at address our of range");
                match points_to {
                    OfficeTile::Number(num) => {
                        if num < 0 {
                            panic!("Cannot jump to negative address")
                        }
                        num as usize
                    }
                    OfficeTile::Character(_) => panic!("Character cannot be used as address"),
                }
            }
            Address::Address(addr) => *addr,
        }
    }
}

enum RuntimeError {
    EmptyHandsError(DebugInfo, Instruction),
    EmptyTileError(DebugInfo, Instruction),
    OverflowError(DebugInfo, Instruction),
    TypeError(DebugInfo, Instruction),
}

enum ArithmeticError {
    OverflowError,
    TypeError,
}

impl fmt::Debug for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RuntimeError::EmptyHandsError(info, instr) => {
                writeln!(f, "Can't do {} - your hands are empty!", instr)?;
                writeln!(f, "line: {}", info.line)
            }
            RuntimeError::EmptyTileError(info, instr) => {
                writeln!(f, "Can't {} - the tile is empty!", instr)?;
                writeln!(f, "line: {}", info.line)
            }
            RuntimeError::OverflowError(info, instr) => {
                writeln!(f, "Operation: {} overflowed", instr)?;
                writeln!(f, "line: {}", info.line)
            }
            RuntimeError::TypeError(info, instr) => {
                writeln!(f, "can't {} - Incompatible types", instr)?;
                writeln!(f, "line: {}", info.line)
            }
        }
    }
}

trait Executable {
    fn execute(&self, state: &mut OfficeState, debug: &DebugInfo) -> Result<bool, RuntimeError>;
}

fn arithmetic_to_runtime_error(
    val: Result<OfficeTile, ArithmeticError>,
    instr: &Instruction,
    debug: DebugInfo,
) -> Result<OfficeTile, RuntimeError> {
    match val {
        Ok(num) => Ok(num),
        Err(err) => match err {
            ArithmeticError::OverflowError => {
                return Err(RuntimeError::OverflowError(debug, instr.clone()))
            }
            ArithmeticError::TypeError => {
                return Err(RuntimeError::TypeError(debug, instr.clone()))
            }
        },
    }
}

impl Executable for Instruction {
    fn execute(&self, state: &mut OfficeState, debug: &DebugInfo) -> Result<bool, RuntimeError> {
        let held = state.held;
        let floor = &state.floor;
        let debug = debug.clone();
        match self {
            Instruction::Add(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    None => return Err(RuntimeError::EmptyTileError(debug, self.clone())),
                    Some(val) => match held {
                        None => return Err(RuntimeError::EmptyHandsError(debug, self.clone())),
                        Some(held) => {
                            let res =
                                arithmetic_to_runtime_error(held.checked_add(val), self, debug)?;
                            state.held = Some(res);
                            return Ok(false);
                        }
                    },
                }
            }
            Instruction::Sub(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    None => return Err(RuntimeError::EmptyTileError(debug, self.clone())),
                    Some(val) => match held {
                        None => return Err(RuntimeError::EmptyHandsError(debug, self.clone())),
                        Some(held) => {
                            let res =
                                arithmetic_to_runtime_error(held.checked_sub(val), self, debug)?;
                            state.held = Some(res);
                            return Ok(false);
                        }
                    },
                }
            }
            Instruction::BumpUp(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    Some(val) => {
                        let one = OfficeTile::Number(1);
                        let res = arithmetic_to_runtime_error(val.checked_add(one), self, debug)?;
                        state.floor[addr] = Some(res);
                        state.held = Some(res);
                        return Ok(false);
                    }
                    None => Err(RuntimeError::EmptyTileError(debug, self.clone())),
                }
            }
            Instruction::BumpDown(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    Some(val) => {
                        let one = OfficeTile::Number(1);
                        let res = arithmetic_to_runtime_error(val.checked_sub(one), self, debug)?;
                        state.floor[addr] = Some(res);
                        state.held = Some(res);
                        Ok(false)
                    }
                    None => Err(RuntimeError::EmptyTileError(debug, self.clone())),
                }
            }
            Instruction::CopyFrom(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    Some(val) => {
                        state.held = Some(val);
                        Ok(false)
                    }
                    None => Err(RuntimeError::EmptyTileError(debug, self.clone())),
                }
            }
            Instruction::CopyTo(addr) => match held {
                Some(val) => {
                    let addr = addr.get_value(state);
                    state.floor[addr] = Some(val);
                    Ok(false)
                }
                None => Err(RuntimeError::EmptyHandsError(debug, self.clone())),
            },
            Instruction::Inbox => match state.inbox.pop_front() {
                Some(val) => {
                    println!("Fetching {:?} from inbox", val);
                    println!("Inbox:{:?}", state.inbox);
                    state.held = Some(val);
                    Ok(false)
                }
                None => Ok(true),
            },
            Instruction::Outbox => match held {
                Some(val) => {
                    state.outbox.push_front(val);
                    state.held = None;
                    Ok(false)
                }
                None => Err(RuntimeError::EmptyHandsError(debug, self.clone())),
            },
            Instruction::LabelDef(_) => Ok(false),
            Instruction::Jump(_) => Ok(false),
            Instruction::JumpN(_) => Ok(false),
            Instruction::JumpZ(_) => Ok(false),
            Instruction::Define(_) => Ok(false),
        }
    }
}

fn tokenize_hrm(file: File) -> std::io::Result<(Vec<TokenDebug>)> {
    let reader = BufReader::new(file);
    let mut lines = reader.lines().enumerate();
    {
        //Ensure program starts with the proper header
        let header = "-- HUMAN RESOURCE MACHINE PROGRAM --";
        let (_line_number, line) = lines.next().expect("File has 0 lines");
        let line = line?;
        if line != header {
            eprintln!("File should start with \"{}\" got \"{}\"", header, line);
            panic!("File is not human resource machine file");
        }
    }

    let mut tokens_vec: Vec<TokenDebug> = Vec::new();
    while let Some((line_number, line)) = lines.next() {
        let line = line?;
        let mut tokens = line.split_whitespace();
        while let Some(token) = tokens.next() {
            let new_token: Token = match token {
                "INBOX" => Token::Op(Op::Inbox),
                "OUTBOX" => Token::Op(Op::Outbox),
                "COPYFROM" => Token::Op(Op::CopyFrom),
                "COPYTO" => Token::Op(Op::CopyTo),
                "JUMP" => Token::Op(Op::Jump),
                "JUMPZ" => Token::Op(Op::JumpZ),
                "JUMPN" => Token::Op(Op::JumpN),
                "BUMPUP" => Token::Op(Op::BumpUp),
                "BUMPDN" => Token::Op(Op::BumpDown),
                "SUB" => Token::Op(Op::Sub),
                "ADD" => Token::Op(Op::Add),
                "DEFINE" => {
                    let token = tokens.next();
                    let num = tokens.next().expect("Define has no number");
                    let num = num
                        .parse::<usize>()
                        .expect("DEFINE must be followed by a number");
                    let mut svg = String::new();
                    while let Some((_line_number, line)) = lines.next() {
                        let line = line?;
                        if line.ends_with(";") {
                            svg.push_str(&line[..line.len() - 1]); //exclude trailing semicolon
                            break;
                        }
                        svg.push_str(&line[..]);
                    }
                    match token {
                        Some("COMMENT") => Token::Op(Op::Define(Define::Comment(num, svg))),
                        Some("LABEL") => Token::Op(Op::Define(Define::Label(num, svg))),
                        Some(other) => {
                            eprintln!("Expected COMMENT or LABEL after DEFINE, got {}", other);
                            panic!("Tokenization Error")
                        }
                        None => {
                            eprintln!("Expected COMMENT or LABEL after DEFINE,");
                            panic!("Tokenization Error")
                        }
                    }
                }
                label if label.ends_with(':') => {
                    if label.len() == 1 {
                        //invalid - the empty label is not a label
                        panic!("invalid label at line {}", line_number);
                    }
                    let label_name = label[0..label.len() - 1].to_string();
                    Token::Op(Op::LabelDef(label_name))
                }
                address if address.starts_with('[') && address.ends_with(']') => {
                    let address = address.split(|c| c == '[' || c == ']').nth(1).unwrap();
                    let address = address.parse::<usize>().unwrap();
                    Token::Address(Address::AddressOf(address))
                }
                address if address.parse::<usize>().is_ok() => {
                    let address = address.parse::<usize>().unwrap();
                    Token::Address(Address::Address(address))
                }
                label => Token::Label(Label {
                    name: String::from(label),
                }),
            };
            tokens_vec.push(TokenDebug {
                token: new_token,
                debug_info: DebugInfo { line: line_number },
            })
        }
    }
    Ok(tokens_vec)
}

fn tokens_to_instructions(tokens: Vec<TokenDebug>) -> Vec<InstructionDebug> {
    let mut instrs = Vec::new();
    let mut tokens = tokens.into_iter();
    while let Some(token) = tokens.next() {
        let debg = token.debug_info;
        let token = token.token;

        let instr: Instruction = match token {
            Token::Op(op) => match op {
                Op::Inbox => Instruction::Inbox,
                Op::Outbox => Instruction::Outbox,
                Op::Define(def) => Instruction::Define(def),
                Op::CopyFrom | Op::CopyTo | Op::BumpUp | Op::BumpDown | Op::Add | Op::Sub => {
                    let next = &tokens.next().expect("op requires address argument");
                    let next = &next.token;
                    if let Token::Address(addr) = next {
                        match op {
                            Op::CopyFrom => Instruction::CopyFrom(*addr),
                            Op::CopyTo => Instruction::CopyTo(*addr),
                            Op::BumpUp => Instruction::BumpUp(*addr),
                            Op::BumpDown => Instruction::BumpDown(*addr),
                            Op::Add => Instruction::Add(*addr),
                            Op::Sub => Instruction::Sub(*addr),
                            _ => panic!("Interpreter error, case not covered"),
                        }
                    } else {
                        panic!(format!("Expected address, found {:?}", next))
                    }
                }
                Op::Jump | Op::JumpN | Op::JumpZ => {
                    let next = &tokens.next().expect("op requires address argument");
                    let next = &next.token;
                    match next {
                        Token::Label(label) => match op {
                            Op::Jump => Instruction::Jump(label.clone()),
                            Op::JumpN => Instruction::JumpN(label.clone()),
                            Op::JumpZ => Instruction::JumpZ(label.clone()),
                            _ => panic!("Interpreter error, case not covered"),
                        },
                        _ => panic!(format!("Expected address, found {:?}", next)),
                    }
                }
                Op::LabelDef(label) => Instruction::LabelDef(label),
            },
            Token::Address(_address) => {
                eprintln!("{:?}", debg);
                panic!("Address requires op taking address")
            }
            Token::Label(_label) => {
                eprintln!("{:?}", debg);
                panic!("Label requires op taking label")
            }
        };
        instrs.push(InstructionDebug(instr, debg));
    }
    instrs
}

/// Return hashmap of token indicies associated with labels
fn process_labels(instructions: Vec<Instruction>) -> HashMap<String, usize> {
    let mut label_map: HashMap<String, usize> = HashMap::new();
    for (instr_ptr, instruction) in instructions.iter().enumerate() {
        if let Instruction::LabelDef(name) = instruction {
            label_map.insert(name.clone(), instr_ptr);
        }
    }
    label_map
}

fn interpret(instructions: Vec<InstructionDebug>, state: &mut OfficeState) {
    let jmp_map = process_labels(
        instructions
            .iter()
            .map(|instr_debug| instr_debug.0.clone())
            .collect(),
    );
    let mut instr_ptr = 0_usize;
    while instr_ptr < instructions.len() {
        println!("{}", state);
        let instruction = &instructions[instr_ptr];
        println!("Executing {}", instruction);
        let finished = instruction.0.execute(state, &instruction.1).unwrap();
        if finished {
            println!("Finished running program");
            break;
        }
        if let Some(index) = calc_jump(&instruction.0, &jmp_map, state.held) {
            instr_ptr = index;
        } else {
            instr_ptr += 1;
        }
    }
}

fn calc_jump(
    instruction: &Instruction,
    jmp_map: &HashMap<String, usize>,
    held: Option<OfficeTile>,
) -> Option<usize> {
    match instruction {
        Instruction::Jump(label) => {
            let line = jmp_map[&label.name];
            return Some(line);
        }
        Instruction::JumpN(label) => {
            let held_val = held.expect("Cannot JUMPN with empty hands");
            if let OfficeTile::Number(num) = held_val {
                if num < 0 {
                    let line = jmp_map[&label.name];
                    return Some(line);
                }
            }
        }
        Instruction::JumpZ(label) => {
            let held_val = held.expect("Cannot JUMPZ with empty hands");
            if let OfficeTile::Number(num) = held_val {
                if num == 0 {
                    let line = jmp_map[&label.name];
                    return Some(line);
                }
            }
        }
        _ => {}
    }
    None
}

fn run(file: File, mut state: OfficeState) -> std::io::Result<()> {
    let tokens = tokenize_hrm(file)?;
    let instructions = tokens_to_instructions(tokens);
    interpret(instructions, &mut state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::interpret;
    use crate::tokenize_hrm;
    use crate::tokens_to_instructions;
    use crate::OfficeState;
    use crate::OfficeTile;
    use std::collections::VecDeque;
    use std::fs::File;
    #[test]
    fn test_reverse_string() {
        let file = File::open("example.hrm").unwrap();
        let tokens = tokenize_hrm(file).unwrap();
        // for tok in tokens.iter() {
        //     println!("{:?}", tok);
        // }
        let instructions = tokens_to_instructions(tokens);
        // for instruction in instructions.iter() {
        //     println!("{:?}", instruction);
        // }
        let inbox = create_inbox!(
            'b', 'r', 'a', 'i', 'n', 0, 'x', 'y', 0, 'a', 'b', 's', 'e', 'n', 't', 'm', 'i', 'n',
            'd', 'e', 'd', 0
        );
        let floor = create_floor!(15, 14, tile!(0));
        let mut office_state = OfficeState::new_with_inbox_floor(inbox, floor);
        interpret(instructions, &mut office_state);

        let expected_output = create_inbox!(
            'a', 'b', 's', 'e', 'n', 't', 'm', 'i', 'n', 'd', 'e', 'd', 'x', 'y', 'b', 'r', 'a',
            'i', 'n'
        );
        assert_eq!(expected_output, office_state.outbox);
    }
}

fn main() -> std::io::Result<()> {
    let file = File::open("example.hrm")?;
    let inbox = create_inbox!('b', 'r', 'a', 'i', 'n', 0);
    let floor = create_floor!(15, 14, tile!(0));

    let office_state = OfficeState::new_with_inbox_floor(inbox, floor);
    run(file, office_state)?;
    Ok(())
}
