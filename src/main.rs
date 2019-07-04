#![deny(warnings)]
#![deny(clippy::all)]
use std::error::Error;
use std::fs::File;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;


use std::convert::From;
use std::convert::TryFrom;
use std::fmt;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
#[cfg(test)]
use quickcheck::{Arbitrary, Gen};

/// An instruction without it argument
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
/// `OfficeTile`s come in the inbox, are placed on the floor and go out the outbox
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OfficeTile {
    Number(i16),     //numbers in human resource machine are in -999..=999
    Character(char), //chars in human resource machine appear to be all [a-zA-Z]
}

impl From<char> for OfficeTile {
    fn from(c: char) -> Self {
        OfficeTile::Character(c)
    }
}

impl From<i8> for OfficeTile {
    fn from(n: i8) -> Self {
        OfficeTile::Number(i16::from(n))
    }
}

impl TryFrom<i16> for OfficeTile {

    type Error = ArithmeticError;

    fn try_from(n: i16) -> Result<Self, Self::Error> {
        if Self::get_range().contains(&n) {
            Ok(OfficeTile::Number(n))
        } else {
            Err(ArithmeticError::Overflow)
        }

    }
}

impl From<&OfficeTile> for OfficeTile {
    fn from(tile: &Self) -> Self {
        *tile
    }
}

macro_rules! create_inbox {
    () => { //avoids warnings for unused_mut when adding zero elements
        {
            VecDeque::<OfficeTile>::new()
        }
    };
    ( $( $x:expr ),* ) => {
        {
            let mut inbox:VecDeque<OfficeTile> = VecDeque::new();
            $(
                inbox.push_back(OfficeTile::from($x));
            )*
            inbox
        }
    }
}

macro_rules! create_floor {
    ( $( $tile:expr ),* ) => {
        {
            let mut floor:Vec<Option<OfficeTile>> = Vec::new();
            $(
                floor.push(Some(tile!($tile)));
            )*
            floor
        }
    };
    ( len $len:expr, $($index:expr, $tile:expr ),* ) => {
        {
            let mut floor:Vec<Option<OfficeTile>> = Vec::with_capacity($len);
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

#[cfg(test)]
impl Arbitrary for OfficeTile {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        //random number in range [-999,999]
        let num = i16::arbitrary(g) % 999;
        Self::try_from(num).unwrap()
    }
}

impl OfficeTile {
    #[inline]
    fn get_range() -> std::ops::Range<i16> {
        (-999..1000)
    }

    fn checked_add(self, rhs: Self) -> Result<Self, ArithmeticError> {
        match (self, rhs) {
            (OfficeTile::Number(lhs), OfficeTile::Number(rhs)) => match lhs.checked_add(rhs) {
                Some(sum) => {
                    if Self::get_range().contains(&sum) {
                        Ok(OfficeTile::Number(sum))
                    } else {
                        Err(ArithmeticError::Overflow)
                    }
                }
                None => Err(ArithmeticError::Overflow),
            },
            (OfficeTile::Character(lhs), OfficeTile::Character(rhs)) => {
                match (lhs as i16).checked_add(rhs as i16) {
                    Some(sum) => {
                        if Self::get_range().contains(&sum) {
                            Ok(OfficeTile::Number(sum))
                        } else {
                            Err(ArithmeticError::Overflow)
                        }
                    }
                    None => Err(ArithmeticError::Overflow),
                }
            }
            (OfficeTile::Number(_), OfficeTile::Character(_))
            | (OfficeTile::Character(_), OfficeTile::Number(_)) => Err(ArithmeticError::TypeError),
        }
    }

    fn checked_sub(self, rhs: Self) -> Result<Self, ArithmeticError> {
        match (self, rhs) {
            (OfficeTile::Number(lhs), OfficeTile::Number(rhs)) => match lhs.checked_sub(rhs) {
                Some(diff) => {
                    if Self::get_range().contains(&diff) {
                        Ok(OfficeTile::Number(diff))
                    } else {
                        Err(ArithmeticError::Overflow)
                    }
                }
                None => Err(ArithmeticError::Overflow),
            },
            (OfficeTile::Character(lhs), OfficeTile::Character(rhs)) => {
                match (lhs as i16).checked_sub(rhs as i16) {
                    Some(diff) => {
                        if Self::get_range().contains(&diff) {
                            Ok(OfficeTile::Number(diff))
                        } else {
                            Err(ArithmeticError::Overflow)
                        }
                    }
                    None => Err(ArithmeticError::Overflow),
                }
            }
            (OfficeTile::Number(_), OfficeTile::Character(_))
            | (OfficeTile::Character(_), OfficeTile::Number(_)) => Err(ArithmeticError::TypeError),
        }
    }
}

// TODO: add instruction pointer here
/// The state of the entire office
/// Composed of the tile held by the player, the inbox, outbox and floor
#[derive(Debug, Clone)]
struct OfficeState {
    held: Option<OfficeTile>,
    inbox: VecDeque<OfficeTile>,
    outbox: VecDeque<OfficeTile>,
    floor: Vec<Option<OfficeTile>>,
}

impl OfficeState {
    pub fn new(inbox: VecDeque<OfficeTile>, floor: Vec<Option<OfficeTile>>) -> Self {
        Self {
            held: None,
            inbox,
            outbox: vec![].into_iter().collect(),
            floor,
        }
    }
}

impl fmt::Display for OfficeState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Self {
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
                        usize::try_from(num).unwrap()
                    }
                    OfficeTile::Character(_) => panic!("Character cannot be used as address"),
                }
            }
            Address::Address(addr) => *addr,
        }
    }
}

#[derive(Debug)]
enum RuntimeError {
    EmptyHands(DebugInfo, Instruction),
    EmptyTile(DebugInfo, Instruction),
    Overflow(DebugInfo, Instruction),
    TypeError(DebugInfo, Instruction),
}

#[derive(PartialEq, Eq, Debug)]
enum ArithmeticError {
    Overflow,
    TypeError,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RuntimeError::EmptyHands(info, instr) => {
                writeln!(f, "Can't do {} - your hands are empty!", instr)?;
                writeln!(f, "line: {}", info.line)
            }
            RuntimeError::EmptyTile(info, instr) => {
                writeln!(f, "Can't {} - the tile is empty!", instr)?;
                writeln!(f, "line: {}", info.line)
            }
            RuntimeError::Overflow(info, instr) => {
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

impl Error for RuntimeError {}

impl Error for ArithmeticError {}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArithmeticError::Overflow => write!(f, "Overflow Error. Number exceeded allowed range"),
            ArithmeticError::TypeError => write!(f, "Type Error. Cannot operate on those operands"),
        }
    }
}

impl TryFrom<RuntimeError> for ArithmeticError {
    type Error = ();

    fn try_from(e: RuntimeError) -> Result<Self, ()> {
        match e {
            RuntimeError::Overflow(_, _) => Ok(ArithmeticError::Overflow),
            RuntimeError::TypeError(_, _) => Ok(ArithmeticError::TypeError),
            _ => Err(()),
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
            ArithmeticError::Overflow => Err(RuntimeError::Overflow(debug, instr.clone())),
            ArithmeticError::TypeError => Err(RuntimeError::TypeError(debug, instr.clone())),
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
                    None => Err(RuntimeError::EmptyTile(debug, self.clone())),
                    Some(val) => match held {
                        None => Err(RuntimeError::EmptyHands(debug, self.clone())),
                        Some(held) => {
                            let res =
                                arithmetic_to_runtime_error(held.checked_add(val), self, debug)?;
                            state.held = Some(res);
                            Ok(false)
                        }
                    },
                }
            }
            Instruction::Sub(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    None => Err(RuntimeError::EmptyTile(debug, self.clone())),
                    Some(val) => match held {
                        None => Err(RuntimeError::EmptyHands(debug, self.clone())),
                        Some(held) => {
                            let res =
                                arithmetic_to_runtime_error(held.checked_sub(val), self, debug)?;
                            state.held = Some(res);
                            Ok(false)
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
                        Ok(false)
                    }
                    None => Err(RuntimeError::EmptyTile(debug, self.clone())),
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
                    None => Err(RuntimeError::EmptyTile(debug, self.clone())),
                }
            }
            Instruction::CopyFrom(addr) => {
                let addr = addr.get_value(state);
                match floor[addr] {
                    Some(val) => {
                        state.held = Some(val);
                        Ok(false)
                    }
                    None => Err(RuntimeError::EmptyTile(debug, self.clone())),
                }
            }
            Instruction::CopyTo(addr) => match held {
                Some(val) => {
                    let addr = addr.get_value(state);
                    state.floor[addr] = Some(val);
                    Ok(false)
                }
                None => Err(RuntimeError::EmptyHands(debug, self.clone())),
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
                    state.outbox.push_back(val);
                    state.held = None;
                    Ok(false)
                }
                None => Err(RuntimeError::EmptyHands(debug, self.clone())),
            },
            Instruction::LabelDef(_)
            | Instruction::Jump(_)
            | Instruction::JumpN(_)
            | Instruction::JumpZ(_)
            | Instruction::Define(_) => Ok(false),
        }
    }
}

fn tokenize_hrm(read: &mut Read) -> Result<Vec<TokenDebug>, Box<Error>> {
    let reader = BufReader::new(read);
    let mut lines = reader.lines().enumerate();
    {
        //Ensure program starts with the proper header
        let expected_header = "-- HUMAN RESOURCE MACHINE PROGRAM --";
        let (_line_number, first_line) = lines.next().expect("File has 0 lines");
        let first_line = first_line?;
        if first_line != expected_header {
            eprintln!(
                "File should start with \"{}\" got \"{}\"",
                expected_header, first_line
            );
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
                    let token_type = tokens.next();
                    let num = tokens.next().expect("Define has no number");
                    let num: usize = num.parse()?;
                    let mut svg = String::new();
                    {
                        let mut svg_line: String;
                        while {
                            let (_line_number, line) = lines.next().unwrap();
                            svg_line = line?;
                            !svg_line.ends_with(';')
                        } {
                            svg.push_str(&svg_line[..]);
                        }
                        svg.push_str(&svg_line[..svg_line.len() - 1]); //exclude trailing semicolon
                    }

                    match token_type {
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
fn process_labels(instructions: &[Instruction]) -> HashMap<String, usize> {
    let mut label_map: HashMap<String, usize> = HashMap::new();
    for (instr_ptr, instruction) in instructions.iter().enumerate() {
        if let Instruction::LabelDef(name) = instruction {
            label_map.insert(name.clone(), instr_ptr);
        }
    }
    label_map
}

fn interpret(
    instructions: &[InstructionDebug],
    state: &mut OfficeState,
) -> Result<(), RuntimeError> {
    let jmp_map = process_labels(
        &instructions
            .iter()
            .map(|instr_debug| instr_debug.0.clone())
            .collect::<Vec<Instruction>>(),
    );
    let mut instr_ptr: usize = 0;
    while instr_ptr < instructions.len() {
        println!("{}", state);
        let InstructionDebug {
            0: instruction,
            1: debug,
        } = &instructions[instr_ptr];
        println!("Executing {}", instruction);
        let finished = instruction.execute(state, &debug)?;
        if finished {
            println!("Finished running program");
            break;
        }
        if let Some(index) = calc_jump(&instruction, &jmp_map, state.held) {
            instr_ptr = index;
        } else {
            instr_ptr += 1;
        }
    }
    Ok(())
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

fn run(read: &mut Read, state: &mut OfficeState) -> Result<(), Box<Error>> {
    let tokens = tokenize_hrm(read)?;
    let instructions = tokens_to_instructions(tokens);
    interpret(&instructions, state)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        interpret, run, tokenize_hrm, tokens_to_instructions, ArithmeticError, OfficeState,
        OfficeTile, RuntimeError,
    };


    use std::collections::VecDeque;
    use std::convert::TryFrom;
    use std::error::Error;
    use std::fs::File;
    use std::path::Path;

    static SOLUTIONS_PATH: &'static str = "./human-resource-machine-solutions";

    macro_rules! test_file {
        ( $filename:expr ) => {
            File::open(Path::new(&format!("{}/{}", SOLUTIONS_PATH, $filename)));
        };
    }

    #[test]
    fn test_01_mail_room() -> Result<(), Box<Error>> {
        let mut file = test_file!("01-Mail-Room.size.speed.asm")?;
        let inbox = create_inbox!(7, 1, 3);
        let expected_out = create_inbox!(7, 1, 3);
        let floor = create_floor!(len 0,);
        let mut office_state = OfficeState::new(inbox, floor);
        run(&mut file, &mut office_state)?;
        assert_eq!(expected_out, office_state.outbox);
        Ok(())
    }

    #[quickcheck]
    fn quickcheck_01_mail_room(inbox0: OfficeTile, inbox1: OfficeTile, inbox2: OfficeTile) -> bool {
        let mut file = test_file!("01-Mail-Room.size.speed.asm").unwrap();
        let initial_inbox = create_inbox!(inbox0, inbox1, inbox2);
        let mut office_state = OfficeState::new(initial_inbox.clone(), create_floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        initial_inbox == office_state.outbox
    }

    #[quickcheck]
    fn quickcheck_02_busy_mail_room_size(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("02-Busy-Mail-Room.size.asm").unwrap();
        let mut office_state = OfficeState::new(inbox.clone(), create_floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        inbox == office_state.outbox
    }

    #[quickcheck]
    fn quickcheck_02_busy_mail_room_speed(mut inbox: VecDeque<OfficeTile>) -> bool {
        //this human resource machine program assumes 12 or fewer inbox items
        let max_size = 12;
        inbox.truncate(max_size);
        let mut file = test_file!("02-Busy-Mail-Room.speed.asm").unwrap();
        let mut office_state = OfficeState::new(inbox.clone(), create_floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        inbox == office_state.outbox
    }

    #[quickcheck]
    fn quickcheck_03_copy_floor(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("03-Copy-Floor.size.speed.asm").unwrap();
        let mut office_state = OfficeState::new(inbox, create_floor!('U', 'J', 'X', 'G', 'B', 'E'));
        run(&mut file, &mut office_state).unwrap();
        create_inbox!('B', 'U', 'G') == office_state.outbox
    }

    #[quickcheck]
    fn quickcheck_04_scrambler_handler(mut inbox: VecDeque<OfficeTile>) -> bool {
        inbox.truncate(inbox.len() / 2 * 2);
        let mut file = test_file!("04-Scrambler-Handler.size.speed.asm").unwrap();
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        let floor = create_floor!(len 3,);
        let mut first_office_state = OfficeState::new(inbox.clone(), floor.clone());
        interpret(&instructions, &mut first_office_state).unwrap();
        let mut office_state = OfficeState::new(first_office_state.outbox.clone(), floor.clone());
        interpret(&instructions, &mut office_state).unwrap();
        inbox == office_state.outbox && first_office_state.outbox == pairwise_reverse(&inbox)
    }

    fn pairwise_reverse(inbox: &VecDeque<OfficeTile>) -> VecDeque<OfficeTile> {
        let mut outbox = VecDeque::new();
        let mut iter = inbox.iter().peekable();
        while iter.peek().is_some() {
            let this = iter.next().unwrap();
            match iter.next() {
                Some(next) => {
                    outbox.push_back(*next);
                    outbox.push_back(*this);
                }
                None => outbox.push_back(*this),
            }
        }
        outbox
    }

    #[quickcheck]
    fn quickcheck_06_rainy_summer(mut inbox: VecDeque<OfficeTile>) -> bool {
        inbox.truncate(inbox.len() / 2 * 2);
        let mut file = test_file!("06-Rainy-Summer.size.speed.asm").unwrap();
        let floor = create_floor!(len 3,);
        let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
        let res = run(&mut file, &mut office_state);
        let expected = pairwise_sum(&inbox);
        match (res, expected) {
            (Err(boxed_err), Err(arith_err)) => match boxed_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (Ok(_), Ok(expected)) => expected == office_state.outbox,
            _ => false,
        }
    }

    fn pairwise_sum(inbox: &VecDeque<OfficeTile>) -> Result<VecDeque<OfficeTile>, ArithmeticError> {
        let mut outbox = VecDeque::new();
        let mut iter = inbox.iter().peekable();
        while iter.peek().is_some() {
            let this = iter.next().unwrap();
            match iter.next() {
                Some(next) => {
                    let res = this.checked_add(*next)?;
                    outbox.push_back(res)
                }
                None => outbox.push_back(*this),
            }
        }
        Ok(outbox)
    }

    #[quickcheck]
    fn quickcheck_07_zero_exterminator(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("07-Zero-Exterminator.size.speed.asm").unwrap();
        let floor = create_floor!(len 9,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        interpret(&instructions, &mut office_state).unwrap();
        let first_office_state = office_state.clone();
        interpret(&instructions, &mut office_state).unwrap();
        first_office_state.outbox == office_state.outbox
            && eliminate_zeroes(&inbox) == first_office_state.outbox
    }

    fn eliminate_zeroes(inbox: &VecDeque<OfficeTile>) -> VecDeque<OfficeTile> {
        let mut res: VecDeque<OfficeTile> = VecDeque::new();
        for tile in inbox {
            if *tile != tile!(0) {
                res.push_back(*tile)
            }
        }
        res
    }

    #[quickcheck]
    fn quickcheck_08_tripler_room(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("08-Tripler-Room.size.speed.asm").unwrap();
        let floor = create_floor!(len 3,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let res = run(&mut file, &mut office_state);
        let expected = triple(&inbox);
        match (res, expected) {
            (Ok(_), Ok(expected_out)) => office_state.outbox == expected_out,
            (Err(boxed_err), Err(arith_err)) => match boxed_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (_, _) => false,
        }
    }

    fn triple(inbox: &VecDeque<OfficeTile>) -> Result<VecDeque<OfficeTile>, ArithmeticError> {
        let mut res = VecDeque::new();
        for tile in inbox {
            let double = tile.checked_add(*tile)?;
            let triple = double.checked_add(*tile)?;
            res.push_back(triple)
        }
        Ok(res)
    }

    #[test]
    fn test_08_tripler_room() -> Result<(), Box<Error>> {
        let mut file = test_file!("08-Tripler-Room.size.speed.asm")?;
        let tokens = tokenize_hrm(&mut file)?;
        let instructions = tokens_to_instructions(tokens);
        let floor = create_floor!(len 3,);
        {
            //test zero
            let inbox = create_inbox!();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = triple(&inbox)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test one
            let inbox = create_inbox!(1);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, create_inbox!(3));
        };
        {
            //test many
            let inbox: VecDeque<OfficeTile> = (0..100).map(OfficeTile::from).collect();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = triple(&inbox)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test max
            let mut inbox: VecDeque<OfficeTile> = VecDeque::new();
            let one_third_of_max = OfficeTile::try_from(333_i16)?;
            inbox.push_back(one_third_of_max);
            let mut expected: VecDeque<OfficeTile> = VecDeque::new();
            let max = OfficeTile::try_from(999_i16)?;
            expected.push_back(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test min
            let mut inbox: VecDeque<OfficeTile> = VecDeque::new();
            let one_third_of_max = OfficeTile::try_from(-333_i16)?;
            inbox.push_back(one_third_of_max);
            let mut expected: VecDeque<OfficeTile> = VecDeque::new();
            let max = OfficeTile::try_from(-999_i16)?;
            expected.push_back(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, expected);
        };
        Ok(())
    }

    #[quickcheck]
    fn quickcheck_09_zero_preservation_initiative(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("09-Zero-Preservation-Initiative.size.asm").unwrap();
        let floor = create_floor!(len 9,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        interpret(&instructions, &mut office_state).unwrap();
        let first_outbox = office_state.outbox.clone();
        interpret(&instructions, &mut office_state).unwrap();
        first_outbox == office_state.outbox && first_outbox.iter().all(|e| *e == tile!(0))
    }

    #[quickcheck]
    fn quickcheck_10_octoplier_suite(inbox: VecDeque<OfficeTile>) -> bool {
        let mut file = test_file!("10-Octoplier-Suite.size.speed.asm").unwrap();
        let floor = create_floor!(len 5,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let res = run(&mut file, &mut office_state);
        let expected = octoply(&inbox);
        match (res, expected) {
            (Ok(_), Ok(expected_out)) => office_state.outbox == expected_out,
            (Err(boxed_err), Err(arith_err)) => match boxed_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (_, _) => false,
        }
    }

    fn octoply(inbox: &VecDeque<OfficeTile>) -> Result<VecDeque<OfficeTile>, ArithmeticError> {
        let mut res = VecDeque::new();
        for tile in inbox {
            let x2 = tile.checked_add(*tile)?;
            let x4 = x2.checked_add(x2)?;
            let x8 = x4.checked_add(x4)?;
            res.push_back(x8);
        }
        Ok(res)
    }

    #[test]
    fn test_10_octoplier_suite() -> Result<(), Box<Error>> {
        let mut file = test_file!("10-Octoplier-Suite.size.speed.asm")?;
        let tokens = tokenize_hrm(&mut file)?;
        let instructions = tokens_to_instructions(tokens);
        let floor = create_floor!(len 3,);
        {
            //test zero
            let inbox = create_inbox!();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = octoply(&inbox)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test one
            let inbox = create_inbox!(1);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, create_inbox!(8));
        };
        {
            //test many
            let inbox: VecDeque<OfficeTile> = (0..100).map(OfficeTile::from).collect();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = octoply(&inbox)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test max
            let mut inbox: VecDeque<OfficeTile> = VecDeque::new();
            let one_eigth_of_max = tile!(124);
            inbox.push_back(one_eigth_of_max);
            let mut expected: VecDeque<OfficeTile> = VecDeque::new();
            let max = OfficeTile::try_from(992_i16)?;
            expected.push_back(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, expected);
        };
        {
            //test min
            let mut inbox: VecDeque<OfficeTile> = VecDeque::new();
            let one_third_of_max = tile!(-124);
            inbox.push_back(one_third_of_max);
            let mut expected: VecDeque<OfficeTile> = VecDeque::new();
            let max = OfficeTile::try_from(-992_i16)?;
            expected.push_back(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert_eq!(office_state.outbox, expected);
        };
        Ok(())
    }

    #[test]
    fn test_reverse_string() -> Result<(), Box<Error>> {
        let mut file = File::open("example.hrm").unwrap();
        let inbox = create_inbox!(
            'b', 'r', 'a', 'i', 'n', 0, 'x', 'y', 0, 'a', 'b', 's', 'e', 'n', 't', 'm', 'i', 'n',
            'd', 'e', 'd', 0
        );
        let floor = create_floor!(len 15, 14, tile!(0));
        let mut office_state = OfficeState::new(inbox, floor);
        run(&mut file, &mut office_state)?;

        let expected_output = create_inbox!(
            'n', 'i', 'a', 'r', 'b', 'y', 'x', 'd', 'e', 'd', 'n', 'i', 'm', 't', 'n', 'e', 's',
            'b', 'a'
        );
        assert_eq!(
            expected_output, office_state.outbox,
            "Reverse failed! Expected {:?}, got {:?}",
            expected_output, office_state.outbox
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<Error>> {
    let mut file = File::open("example.hrm")?;
    let inbox = create_inbox!('b', 'r', 'a', 'i', 'n', 0);
    let floor = create_floor!(len 15, 14, tile!(0));

    let mut office_state = OfficeState::new(inbox, floor);
    run(&mut file, &mut office_state)?;
    Ok(())
}
