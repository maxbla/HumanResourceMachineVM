use std::error::Error;

use std::collections::HashMap;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;

use std::cmp::min;
use std::convert::From;
use std::convert::TryFrom;
use std::fmt;
use std::string::ToString;

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
///
/// Mostly a one-to-one relation to in-game instructions
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

/// An Instruction that includes debug information.
///
/// The debug information includes information like line numbers
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

/// A definition of a comment or label image that has been serialized
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

impl Address {
    fn get_value(
        &self,
        state: &OfficeState,
        info: &DebugInfo,
        instr: &Instruction,
    ) -> Result<usize, RuntimeError> {
        match self {
            Address::AddressOf(addr) => {
                let addr = state
                    .floor
                    .get(*addr)
                    .ok_or_else(|| RuntimeError::Address(info.clone(), instr.clone()))?;
                let points_to =
                    addr.ok_or_else(|| RuntimeError::EmptyTile(info.clone(), instr.clone()))?;
                match points_to {
                    OfficeTile::Number(num) => usize::try_from(num)
                        .map_err(|_| RuntimeError::Overflow(info.clone(), instr.clone())),
                    OfficeTile::Character(_) => {
                        Err(RuntimeError::TypeError(info.clone(), instr.clone()))
                    }
                }
            }
            Address::Address(addr) => Ok(*addr),
        }
    }
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
pub enum OfficeTile {
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
    type Error = (); //ArithmeticError;

    fn try_from(n: i16) -> Result<Self, ()> {
        if Self::RANGE.contains(&n) {
            Ok(OfficeTile::Number(n))
        } else {
            Err(())
        }
    }
}

/// Macro for generating "inboxes"
///
/// An inbox is just a Vec of OfficeTiles, but it is accessed like a stack, so
/// the ordering is unintuitive. Use this macro to create an inbox that
/// contains OfficeTiles that will be read in the order they are put in.
///
/// # Examples
/// ```
/// use human_resource_machine::{inbox, OfficeTile};
/// let mut inbox = inbox!('a', 5);
/// assert_eq!(inbox.pop(), Some(OfficeTile::Character('a')));
/// assert_eq!(inbox.pop(), Some(OfficeTile::Number(5)));
/// assert_eq!(inbox.pop(), None);
/// ```
#[macro_export]
macro_rules! inbox {
    ( $( $x:expr ),* ) => {
        {
            #[allow(unused_mut)]
            let mut inbox = Vec::<OfficeTile>::new();
            $(
                inbox.push(OfficeTile::from($x));
            )*
            inbox.into_iter().rev().collect::<Vec<OfficeTile>>()
        }
    }
}

/// Macro for generating "outboxes". Used in testing.
///
/// An outbox is just a Vec of OfficeTiles.
macro_rules! outbox {
    ( $( $x:expr ),* ) => {
        {
            #[allow(unused_mut)]
            let mut outbox = Vec::new();
            $(
                outbox.push(OfficeTile::from($x));
            )*
            outbox
        }
    }
}

/// Macro for generating "floors"
///
/// Floors are Vec<Option<OfficeTile>>. Despite the game displaying them as
/// grids, they are indexed linearly.
/// # Examples
/// ```
/// use human_resource_machine::{floor, OfficeTile};
/// let empty_floor = floor!(len 10,);
/// assert_eq!(empty_floor[0], None);
/// let full_floor = floor!('a', 'b', 'c', 'd');
/// assert_eq!(full_floor[3], Some(OfficeTile::from('d')));
/// ```
#[macro_export]
macro_rules! floor { //TODO: make more similar to the vec! macro
    ( $( $tile:expr ),* ) => {
        {
            let mut floor:Vec<Option<OfficeTile>> = Vec::new();
            $(
                floor.push(Some(OfficeTile::from($tile)));
            )*
            floor
        }
    };
    ( len $len:expr, $({$index:expr, $tile:expr }),* ) => {
        {
            let mut floor:Vec<Option<OfficeTile>> = Vec::with_capacity($len);
            for _ in 0..$len {
                floor.push(None)
            }
            $(
                floor[$index] = Some(OfficeTile::from($tile));
            )*
            floor
        }
    }
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

/// Represents a tile in-game.
/// 
/// Tiles are the things that appear in the inbox, are placed in the outbox,
/// can be held by the player and are spread across the floor in Human
/// Resource Machine
impl OfficeTile {
    const RANGE: std::ops::RangeInclusive<i16> = (-999..=999);

    fn checked_add(self, rhs: Self) -> Result<Self, ArithmeticError> {
        let gen_error = || ArithmeticError::Overflow(self, rhs);
        match (self, rhs) {
            (OfficeTile::Number(lhs), OfficeTile::Number(rhs)) => {
                let sum = lhs.checked_add(rhs).ok_or_else(gen_error)?;
                OfficeTile::try_from(sum).map_err(|_| gen_error())
            }
            (lhs, rhs) => {
                Err(ArithmeticError::TypeError(lhs, rhs))
            }
        }
    }

    fn checked_sub(self, rhs: Self) -> Result<Self, ArithmeticError> {
        let gen_error = || ArithmeticError::Overflow(self, rhs);
        match (self, rhs) {
            (OfficeTile::Number(lhs), OfficeTile::Number(rhs)) => {
                let diff = lhs
                    .checked_sub(rhs)
                    .ok_or_else(gen_error)?;
                OfficeTile::try_from(diff).map_err(|_| gen_error())
            }
            (OfficeTile::Character(lhs), OfficeTile::Character(rhs)) => {
                let diff = (lhs as i16)
                    .checked_sub(rhs as i16)
                    .ok_or_else(gen_error)?;
                OfficeTile::try_from(diff).map_err(|_| gen_error())
            }
            (lhs, rhs) => Err(ArithmeticError::TypeError(lhs, rhs)),
        }
    }
}

// TODO: add instruction pointer here
/// The state of the entire office
/// Composed of the tile held by the player, the inbox, the outbox and the
/// floor
#[derive(Debug, Clone)]
pub struct OfficeState {
    held: Option<OfficeTile>,
    /// OfficeTiles that will be inboxed
    /// The highest index tile is the next one to be inboxed
    inbox: Vec<OfficeTile>,
    /// OfficeTiles that have been outboxed
    /// The highest index tile is the one that has been most recently outboxed
    pub outbox: Vec<OfficeTile>,
    floor: Vec<Option<OfficeTile>>,
}

impl OfficeState {
    pub fn new(inbox: Vec<OfficeTile>, floor: Vec<Option<OfficeTile>>) -> Self {
        Self {
            held: None,
            inbox,
            outbox: outbox!(),
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
        let max_rows = 7;
        let num_rows: usize = *vec![
            floor.len() / 5,
            min(inbox.len(), max_rows),
            min(outbox.len(), max_rows),
        ]
        .iter()
        .max()
        .unwrap();

        let floor_width = 5; // number of floor items per row
        let floor_string_width = 4 * floor_width + 1;
        let held_string = format!(
            "held: [{}] ",
            held.map(|h| h.to_string()).unwrap_or_default()
        );
        writeln!(f, "in |{:^21}|out", held_string)?;
        writeln!(f, "---+{:^21}+---", "-".repeat(21))?;
        for row in 0..num_rows {
            let mut s = String::with_capacity(floor_string_width);
            for index in row * floor_width..(row + 1) * floor_width {
                let floor_val = floor
                    .get(index)
                    .map(|val| val.map(|val| val.to_string()))
                    .unwrap_or_default()
                    .unwrap_or_default();
                s.push_str(&format!("{:^4}", floor_val));
            }
            let in_idx = inbox.len().checked_sub(row + 1);
            // TODO: use Option::flatten() once it lands in stable
            let inbox_val = in_idx
                .map(|i| inbox.get(i).map(ToString::to_string))
                .unwrap_or_default()
                .unwrap_or_default();
            let out_idx = outbox.len().checked_sub(row + 1);
            let outbox_val = out_idx
                .map(|i| outbox.get(i).map(ToString::to_string))
                .unwrap_or_default()
                .unwrap_or_default();
            writeln!(f, "{:<3}|{:^21}|{:>3}", inbox_val, s, outbox_val)?
        }
        Ok(())
    }
}

/// Represents a runtime error in human resource machine, something that causes
/// execution to stop and your boss to give you an explaination of the error
#[derive(Debug)]
enum RuntimeError {
    /// CopyTo while OfficeState.held.is_none()
    EmptyHands(DebugInfo, Instruction),
    // CopyFrom floor tile that is_none()
    EmptyTile(DebugInfo, Instruction),
    /// Under or overflow, for example when trying to use a negative number as
    /// an address
    Overflow(DebugInfo, Instruction),
    TypeError(DebugInfo, Instruction),
    Address(DebugInfo, Instruction),
    ArithmeticError(ArithmeticError, DebugInfo, Instruction)
}

/// A runtime error generated by doing invalid arithmetic
#[derive(PartialEq, Eq, Debug)]
pub enum ArithmeticError {
    /// OfficeTile::RANGE.contains(value) is false for the result value
    Overflow(OfficeTile, OfficeTile),
    /// The type rules for valid arithmetic were violated
    /// 
    /// You can add or subtract two numbers and you can subtract two characters
    /// but you can't add characters and you can't do any arithmetic between a
    /// character and a number
    TypeError(OfficeTile, OfficeTile),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RuntimeError::EmptyHands(info, instr) => {
                writeln!(f, "Can't do {} - your hands are empty!", instr)?;
                writeln!(f, "on line: {}", info.line)
            }
            RuntimeError::EmptyTile(info, instr) => {
                writeln!(f, "Can't {} - the tile is empty!", instr)?;
                writeln!(f, "on line: {}", info.line)
            }
            RuntimeError::Overflow(info, instr) => {
                writeln!(f, "Operation: {} overflowed", instr)?;
                writeln!(f, "on line: {}", info.line)
            }
            RuntimeError::TypeError(info, instr) => {
                writeln!(f, "can't {} - Incompatible types", instr)?;
                writeln!(f, "on line: {}", info.line)
            }
            RuntimeError::Address(info, instr) => {
                writeln!(f, "can't {} - Address out of range", instr)?;
                writeln!(f, "on line: {}", info.line)
            }
            RuntimeError::ArithmeticError(err, info, instr) => {
                writeln!(f, "{}", err)?;
                writeln!(f, "on line: {}\nwhile executing:{}", info.line, instr)
            }
        }
    }
}

impl Error for RuntimeError {}

impl Error for ArithmeticError {}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArithmeticError::Overflow(lhs, rhs) => write!(f, "Overflow Error between {} and {}", lhs, rhs),
            ArithmeticError::TypeError(lhs, rhs) => write!(f, "Type Error between {} and {}", lhs, rhs),
        }
    }
}

impl TryFrom<RuntimeError> for ArithmeticError {
    type Error = ();

    fn try_from(e: RuntimeError) -> Result<Self, ()> {
        match e {
            RuntimeError::ArithmeticError(arith_error, _, _) => {Ok(arith_error)}
            _ => Err(()),
        }
    }
}

fn arithmetic_to_runtime_error(
    val: Result<OfficeTile, ArithmeticError>,
    instr: &Instruction,
    debug: DebugInfo,
) -> Result<OfficeTile, RuntimeError> {
    val.map_err(|err| RuntimeError::ArithmeticError(err, debug, instr.clone()))
}

impl Instruction {
    fn execute(&self, state: &mut OfficeState, debug: &DebugInfo) -> Result<bool, RuntimeError> {
        let held = state.held;
        let floor = &state.floor;
        let debug = debug.clone();
        match self {
            Instruction::Add(addr) => {
                let addr = addr.get_value(state, &debug, &self)?;
                let val = floor[addr]
                    .ok_or_else(|| RuntimeError::EmptyTile(debug.clone(), self.clone()))?;
                let held =
                    held.ok_or_else(|| RuntimeError::EmptyHands(debug.clone(), self.clone()))?;
                let res = arithmetic_to_runtime_error(held.checked_add(val), self, debug)?;
                state.held = Some(res);
            }
            Instruction::Sub(addr) => {
                let addr = addr.get_value(state, &debug, &self)?;
                let val = floor[addr]
                    .ok_or_else(|| RuntimeError::EmptyTile(debug.clone(), self.clone()))?;
                let held =
                    held.ok_or_else(|| RuntimeError::EmptyHands(debug.clone(), self.clone()))?;
                let res = arithmetic_to_runtime_error(held.checked_sub(val), self, debug)?;
                state.held = Some(res);
            }
            Instruction::BumpUp(addr) => {
                let addr = addr.get_value(state, &debug, &self)?;
                let val = floor[addr]
                    .ok_or_else(|| RuntimeError::EmptyTile(debug.clone(), self.clone()))?;
                let res = arithmetic_to_runtime_error(val.checked_add(1.into()), self, debug)?;
                state.floor[addr] = Some(res);
                state.held = Some(res);
            }
            Instruction::BumpDown(addr) => {
                let addr = addr.get_value(state, &debug, &self)?;
                let val = floor[addr]
                    .ok_or_else(|| RuntimeError::EmptyTile(debug.clone(), self.clone()))?;
                let res = arithmetic_to_runtime_error(val.checked_sub(1.into()), self, debug)?;
                state.floor[addr] = Some(res);
                state.held = Some(res);
            }
            Instruction::CopyFrom(addr) => {
                let addr = addr.get_value(state, &debug, &self)?;
                let val =
                    floor[addr].ok_or_else(|| RuntimeError::EmptyTile(debug, self.clone()))?;
                state.held = Some(val);
            }
            Instruction::CopyTo(addr) => {
                let val =
                    held.ok_or_else(|| RuntimeError::EmptyHands(debug.clone(), self.clone()))?;
                let addr = addr.get_value(state, &debug, &self)?;
                state.floor[addr] = Some(val);
            }
            Instruction::Inbox => match state.inbox.pop() {
                Some(val) => state.held = Some(val),
                None => return Ok(true),
            },
            Instruction::Outbox => {
                let val = held.ok_or_else(|| RuntimeError::EmptyHands(debug, self.clone()))?;
                state.outbox.push(val);
                state.held = None;
            }
            Instruction::LabelDef(_)
            | Instruction::Jump(_)
            | Instruction::JumpN(_)
            | Instruction::JumpZ(_)
            | Instruction::Define(_) => (), // no op
        }
        Ok(false)
    }
}

fn tokenize_hrm(read: &mut dyn Read) -> Result<Vec<TokenDebug>, Box<dyn Error>> {
    let reader = BufReader::new(read);
    let mut lines = reader.lines().enumerate();
    {
        //Ensure program starts with the proper header
        let expected_header = "-- HUMAN RESOURCE MACHINE PROGRAM --";
        let (_line_number, first_line) = lines.next().expect("File has 0 lines");
        let first_line = first_line?;
        if first_line != expected_header {
            eprintln!(
                "File should start with:\n\"{}\"\nFirst line is:\n\"{}\"",
                expected_header, first_line
            );
            panic!("File is not human resource machine file");
        }
    }

    let mut tokens_vec: Vec<TokenDebug> = Vec::new();
    'outer: while let Some((line_number, line)) = lines.next() {
        let line = line?;
        if line.starts_with("--") && line.ends_with("--") {
            continue;
        }
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
                "COMMENT" => continue 'outer, //TODO: keep track of comments
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
                    let mut label = label.to_string();
                    label.pop(); // remove trailing :
                    if label.is_empty() {
                        panic!("invalid label at line {}", line_number);
                    }
                    Token::Op(Op::LabelDef(label))
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
                            _ => unreachable!("Interpreter error, case not covered"),
                        }
                    } else {
                        panic!("Expected address, found {:?}", next)
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
                            _ => unreachable!("Interpreter error, case not covered"),
                        },
                        _ => panic!("Expected address, found {:?}", next),
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

/// Runs a program on an OfficeState, mutating the OfficeState along the way
/// # Examples
/// ```
/// use human_resource_machine::{OfficeState, inbox, floor, OfficeTile, run};
/// use std::io::Read;
/// let mut state = OfficeState::new(inbox!('a'), floor!());
/// let mut program: &[u8] = "-- HUMAN RESOURCE MACHINE PROGRAM --\nINBOX\nOUTBOX".as_bytes();
/// run(&mut program, &mut state);
/// assert_eq!(state.outbox, vec!(OfficeTile::from('a')));
/// ```
pub fn run(read: &mut dyn Read, state: &mut OfficeState) -> Result<(), Box<dyn Error>> {
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

    use std::convert::TryFrom;
    use std::error::Error;
    use std::fs::File;
    use std::iter::Iterator;
    use std::path::Path;

    use quickcheck::TestResult;

    static SOLUTIONS_PATH: &str = "./human-resource-machine-solutions";

    macro_rules! test_file {
        ( $filename:expr ) => {
            File::open(Path::new(&format!("{}/{}", SOLUTIONS_PATH, $filename)));
        };
    }

    /// Quick shorthand for OfficeTile::from
    macro_rules! tile {
        ( $value:expr ) => {{
            OfficeTile::from($value)
        }};
    }

    /// Equality test between inbox/outbox
    /// 
    /// Inbox and outbox are used as queues, and we want the simple program
    /// INBOX
    /// OUTBOX
    /// to produce a mutually equal inbox and outbox, so reverse one when
    /// comparing for equality
    fn box_eq(inbox: &Vec<OfficeTile>, outbox: &Vec<OfficeTile>) -> bool {
        Iterator::eq(inbox.iter().rev(), outbox.iter())
    }

    #[test]
    fn test_01_mail_room() -> Result<(), Box<dyn Error>> {
        let mut file = test_file!("01-Mail-Room.size.speed.asm")?;
        let inbox = inbox!(7, 1, 3);
        let expected_out = inbox!(7, 1, 3);
        let floor = floor!(len 0,);
        let mut office_state = OfficeState::new(inbox, floor);
        run(&mut file, &mut office_state)?;
        assert!(box_eq(&expected_out, &office_state.outbox));
        Ok(())
    }

    #[quickcheck]
    fn quickcheck_01_mail_room(inbox0: OfficeTile, inbox1: OfficeTile, inbox2: OfficeTile) -> bool {
        let mut file = test_file!("01-Mail-Room.size.speed.asm").unwrap();
        let initial_inbox = inbox!(inbox0, inbox1, inbox2);
        let mut office_state = OfficeState::new(initial_inbox.clone(), floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        box_eq(&initial_inbox, &office_state.outbox)
    }

    #[quickcheck]
    fn quickcheck_02_busy_mail_room_size(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("02-Busy-Mail-Room.size.asm").unwrap();
        let mut office_state = OfficeState::new(inbox.clone(), floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        box_eq(&inbox, &office_state.outbox)
    }

    #[quickcheck]
    fn quickcheck_02_busy_mail_room_speed(mut inbox: Vec<OfficeTile>) -> bool {
        //this human resource machine program assumes 12 or fewer inbox items
        let max_size = 12;
        inbox.truncate(max_size);
        let mut file = test_file!("02-Busy-Mail-Room.speed.asm").unwrap();
        let mut office_state = OfficeState::new(inbox.clone(), floor!(len 0,));
        run(&mut file, &mut office_state).unwrap();
        box_eq(&inbox, &office_state.outbox)
    }

    #[quickcheck]
    fn quickcheck_03_copy_floor(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("03-Copy-Floor.size.speed.asm").unwrap();
        let mut office_state = OfficeState::new(inbox, floor!('U', 'J', 'X', 'G', 'B', 'E'));
        run(&mut file, &mut office_state).unwrap();
        box_eq(&inbox!('B', 'U', 'G'), &office_state.outbox)
    }

    #[quickcheck]
    fn quickcheck_04_scrambler_handler(mut inbox: Vec<OfficeTile>) -> bool {
        inbox.truncate(inbox.len() / 2 * 2);
        let mut file = test_file!("04-Scrambler-Handler.size.speed.asm").unwrap();
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        let floor = floor!(len 3,);
        let mut first_office_state = OfficeState::new(inbox.clone(), floor.clone());
        interpret(&instructions, &mut first_office_state).unwrap();
        let mut office_state = OfficeState::new(first_office_state.outbox.clone(), floor.clone());
        interpret(&instructions, &mut office_state).unwrap();
        &inbox == &office_state.outbox
            && box_eq(&first_office_state.outbox, &pairwise_reverse(&inbox))
    }

    fn pairwise_reverse(inbox: &Vec<OfficeTile>) -> Vec<OfficeTile> {
        inbox
            .chunks_exact(2)
            .flat_map(|chunk| chunk.iter().rev())
            .copied()
            .collect()
    }

    #[quickcheck]
    fn quickcheck_06_rainy_summer(mut inbox: Vec<OfficeTile>) -> bool {
        inbox.truncate(inbox.len() / 2 * 2);
        let mut file = test_file!("06-Rainy-Summer.size.speed.asm").unwrap();
        let floor = floor!(len 3,);
        let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
        let res = run(&mut file, &mut office_state);
        let expected = pairwise_sum(&inbox);
        match (res, expected) {
            (Err(rt_err), Err(arith_err)) => match rt_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (Ok(_), Ok(expected)) => box_eq(&expected, &office_state.outbox),
            _ => false,
        }
    }

    fn pairwise_sum(inbox: &Vec<OfficeTile>) -> Result<Vec<OfficeTile>, ArithmeticError> {
        let mut outbox = Vec::new();
        let mut iter = inbox.iter();
        while let (Some(this), Some(next)) = (iter.next(), iter.next()) {
            outbox.push(this.checked_add(*next)?)
        }
        Ok(outbox)
    }

    #[quickcheck]
    fn quickcheck_07_zero_exterminator(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("07-Zero-Exterminator.size.speed.asm").unwrap();
        let floor = floor!(len 9,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        interpret(&instructions, &mut office_state).unwrap();
        let first_office_state = office_state.clone();
        interpret(&instructions, &mut office_state).unwrap();
        &first_office_state.outbox == &office_state.outbox
            && box_eq(&eliminate_zeroes(&inbox), &first_office_state.outbox)
    }

    fn eliminate_zeroes(inbox: &Vec<OfficeTile>) -> Vec<OfficeTile> {
        let mut res: Vec<OfficeTile> = Vec::new();
        for tile in inbox {
            if *tile != tile!(0) {
                res.push(*tile)
            }
        }
        res
    }

    #[quickcheck]
    fn quickcheck_08_tripler_room(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("08-Tripler-Room.size.speed.asm").unwrap();
        let floor = floor!(len 3,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let res = run(&mut file, &mut office_state);
        let expected = triple(&inbox);
        match (res, expected) {
            (Ok(_), Ok(expected_out)) => box_eq(&office_state.outbox, &expected_out),
            (Err(boxed_err), Err(arith_err)) => match boxed_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (_, _) => false,
        }
    }

    fn triple(inbox: &Vec<OfficeTile>) -> Result<Vec<OfficeTile>, ArithmeticError> {
        let mut res = Vec::new();
        for tile in inbox {
            let double = tile.checked_add(*tile)?;
            let triple = double.checked_add(*tile)?;
            res.push(triple)
        }
        Ok(res)
    }

    #[test]
    fn test_08_tripler_room() -> Result<(), Box<dyn Error>> {
        let mut file = test_file!("08-Tripler-Room.size.speed.asm")?;
        let tokens = tokenize_hrm(&mut file)?;
        let instructions = tokens_to_instructions(tokens);
        let floor = floor!(len 3,);
        {
            //test zero
            let inbox = inbox!();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = triple(&inbox)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test one
            let inbox = inbox!(1);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &inbox!(3)));
        };
        {
            //test many
            let inbox: Vec<OfficeTile> = (0..100).map(OfficeTile::from).collect();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = triple(&inbox)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test max
            let mut inbox: Vec<OfficeTile> = Vec::new();
            let one_third_of_max = OfficeTile::try_from(333_i16).unwrap();
            inbox.push(one_third_of_max);
            let mut expected: Vec<OfficeTile> = Vec::new();
            let max = OfficeTile::try_from(999_i16).unwrap();
            expected.push(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test min
            let mut inbox: Vec<OfficeTile> = Vec::new();
            let one_third_of_max = OfficeTile::try_from(-333_i16).unwrap();
            inbox.push(one_third_of_max);
            let mut expected: Vec<OfficeTile> = Vec::new();
            let max = OfficeTile::try_from(-999_i16).unwrap();
            expected.push(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        Ok(())
    }

    #[quickcheck]
    fn quickcheck_09_zero_preservation_initiative(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("09-Zero-Preservation-Initiative.size.asm").unwrap();
        let floor = floor!(len 9,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        interpret(&instructions, &mut office_state).unwrap();
        let first_outbox = office_state.outbox.clone();
        interpret(&instructions, &mut office_state).unwrap();
        first_outbox == office_state.outbox && first_outbox.iter().all(|e| *e == tile!(0))
    }

    #[quickcheck]
    fn quickcheck_10_octoplier_suite(inbox: Vec<OfficeTile>) -> bool {
        let mut file = test_file!("10-Octoplier-Suite.size.speed.asm").unwrap();
        let floor = floor!(len 5,);
        let mut office_state = OfficeState::new(inbox.clone(), floor);
        let res = run(&mut file, &mut office_state);
        let expected = octoply(&inbox);
        match (res, expected) {
            (Ok(_), Ok(expected_out)) => box_eq(&office_state.outbox, &expected_out),
            (Err(boxed_err), Err(arith_err)) => match boxed_err.downcast::<RuntimeError>() {
                Ok(err) => ArithmeticError::try_from(*err).unwrap() == arith_err,
                _ => false,
            },
            (_, _) => false,
        }
    }

    fn octoply(inbox: &Vec<OfficeTile>) -> Result<Vec<OfficeTile>, ArithmeticError> {
        let mut res = Vec::new();
        for tile in inbox {
            let x2 = tile.checked_add(*tile)?;
            let x4 = x2.checked_add(x2)?;
            let x8 = x4.checked_add(x4)?;
            res.push(x8);
        }
        Ok(res)
    }

    #[test]
    fn test_10_octoplier_suite() -> Result<(), Box<dyn Error>> {
        let mut file = test_file!("10-Octoplier-Suite.size.speed.asm")?;
        let tokens = tokenize_hrm(&mut file)?;
        let instructions = tokens_to_instructions(tokens);
        let floor = floor!(len 3,);
        {
            //test zero
            let inbox = inbox!();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = octoply(&inbox)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test one
            let inbox = inbox!(1);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &inbox!(8)));
        };
        {
            //test many
            let inbox: Vec<OfficeTile> = (0..100).map(OfficeTile::from).collect();
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            let expected = octoply(&inbox)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test max
            let mut inbox: Vec<OfficeTile> = Vec::new();
            let one_eigth_of_max = tile!(124);
            inbox.push(one_eigth_of_max);
            let mut expected: Vec<OfficeTile> = Vec::new();
            let max = OfficeTile::try_from(992_i16).unwrap();
            expected.push(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        {
            //test min
            let mut inbox: Vec<OfficeTile> = Vec::new();
            let one_third_of_max = tile!(-124);
            inbox.push(one_third_of_max);
            let mut expected: Vec<OfficeTile> = Vec::new();
            let max = OfficeTile::try_from(-992_i16).unwrap();
            expected.push(max);
            let mut office_state = OfficeState::new(inbox.clone(), floor.clone());
            interpret(&instructions, &mut office_state)?;
            assert!(box_eq(&office_state.outbox, &expected));
        };
        Ok(())
    }

    #[quickcheck]
    fn quickcheck_31_string_reverse(mut inbox: Vec<OfficeTile>) -> TestResult {
        if inbox.iter().any(|&e| e == tile!(0)) || inbox.len() < 2 {
            return TestResult::discard();
        }
        inbox.truncate(11);
        let orig_inbox = inbox.clone();
        inbox.insert(0, tile!(0));
        let inbox = inbox; // shadow inbox to make it not mutable
        let mut file = test_file!("31-String-Reverse.speed.asm").unwrap();
        let floor = floor!(len 15, {14, 0});
        let mut office_state = OfficeState::new(inbox, floor.clone());
        let tokens = tokenize_hrm(&mut file).unwrap();
        let instructions = tokens_to_instructions(tokens);
        if interpret(&instructions, &mut office_state).is_err() {
            return TestResult::failed();
        }
        office_state.outbox.insert(0, tile!(0));
        let mut office_state = OfficeState::new(office_state.outbox, floor);
        if interpret(&instructions, &mut office_state).is_err() {
            return TestResult::failed();
        }
        TestResult::from_bool(office_state.outbox == orig_inbox)
    }

    #[test]
    fn test_reverse_string() -> Result<(), Box<dyn Error>> {
        let mut file = test_file!("31-String-Reverse.speed.asm").unwrap();
        let inbox = inbox!(
            'b', 'r', 'a', 'i', 'n', 0, 'x', 'y', 0, 'a', 'b', 's', 'e', 'n', 't', 'm', 'i', 'n',
            'd', 'e', 'd', 0
        );
        let floor = floor!(len 15, {14, 0});
        let mut office_state = OfficeState::new(inbox, floor);
        run(&mut file, &mut office_state)?;

        let expected_output = outbox!(
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
