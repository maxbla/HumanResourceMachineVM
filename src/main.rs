use std::fs::File;

use std::io::BufRead;
use std::io::BufReader;

use std::ops::Add;
use std::ops::Sub;

use std::collections::HashMap;
use std::collections::VecDeque;

use std::fmt;

/// An instruction without it argument
#[derive(Debug, Clone, PartialEq, Eq)]
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
    Comment,
    Define, //ignored, defines the shapes of drawings
    Label,
    Svg(String),
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
    Comment,
    Define, //ignored, defines the shapes of drawings
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// label as argument, not definition of a label
struct Label {
    name: String,
}

/// Space separated syntax element
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Op(Op),
    Address(Address),
    Label(Label),
}

/// Token that refers to an office tile on the floor
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Address {
    Address(usize),
    AddressOf(usize),
}

/// A token with some extra debug info
#[derive(Debug)]
struct TokenDebug {
    token: Token,
    debug_info: DebugInfo,
}

/// The debug info of a token e.g. the line it occured in the original source
#[derive(Debug)]
struct DebugInfo {
    line_number: usize,
}

/// A value
/// Can either be a character or a number (integer)
/// OfficeTiles come in the inbox, are placed on the floor and go out the outbox
#[derive(Debug, Copy, Clone)]
enum OfficeTile {
    Number(i16),     //numbers in human resource machine are in -999..=999
    Character(char), //chars in human resource machine appear to be all [a-zA-Z]
}


impl OfficeTile {
    fn from_char(c: char) -> OfficeTile {
        OfficeTile::Character(c)
    }

    fn from_number(n: i16) -> OfficeTile {
        OfficeTile::Number(n)
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

impl Add for OfficeTile {
    type Output = OfficeTile;

    fn add(self, other: OfficeTile) -> OfficeTile {
        match self {
            OfficeTile::Number(num) => match other {
                OfficeTile::Number(other_num) => {
                    let res = num + other_num;
                    if res > 999_i16 {
                        eprintln!("Tried to ADD {} from {}, got {}", other_num, num, res);
                        panic!("Overflow: All numbers must be in range -999..=999");
                    }
                    OfficeTile::Number(res)
                }
                OfficeTile::Character(_) => panic!("Can't ADD between letter and number"),
            },
            OfficeTile::Character(character) => match other {
                OfficeTile::Character(other_char) => {
                    let res: i16 = character as i16 + other_char as i16;
                    if res > 999_i16 {
                        eprintln!(
                            "Tried to ADD {} from {}, got {}",
                            other_char, character, res
                        );
                        panic!("Overflow: All numbers must be in range -999..=999");
                    }
                    OfficeTile::Number(res)
                }
                OfficeTile::Number(_) => panic!("Can't ADD between letter and number"),
            },
        }
    }
}

impl Sub for OfficeTile {
    type Output = OfficeTile;

    fn sub(self, other: OfficeTile) -> OfficeTile {
        match self {
            OfficeTile::Number(num) => match other {
                OfficeTile::Number(other_num) => {
                    let res = num - other_num;
                    if res < -999_i16 {
                        eprintln!("Tried to SUB {} from {}, got {}", other_num, num, res);
                        panic!("Underflow: All numbers must be in range -999..=999");
                    }
                    OfficeTile::Number(res)
                }
                OfficeTile::Character(_) => panic!("Can't SUB between letter and number"),
            },
            OfficeTile::Character(character) => match other {
                OfficeTile::Character(other_char) => {
                    let res: i16 = character as i16 - other_char as i16;
                    if res < -999_i16 {
                        eprintln!(
                            "Tried to SUB {} from {}, got {}",
                            other_char, character, res
                        );
                        panic!("Underflow: All numbers must be in range -999..=999");
                    }
                    OfficeTile::Number(res)
                }
                OfficeTile::Number(_) => panic!("Can't SUB between letter and number"),
            },
        }
    }
}

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
    fn new() -> OfficeState {
        OfficeState::new_with_inbox_floor(vec![].into_iter().collect(), vec![])
    }

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
            self.inbox.len(),
            self.outbox.len(),
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

trait Executable {
    fn execute(&self, state: &mut OfficeState);
}

impl Executable for Instruction {
    fn execute(&self, state: &mut OfficeState) {
        let held = &state.held;
        let floor = &state.floor;
        match self {
            Instruction::Add(addr) => {
                let addr = addr.get_value(state);
                let other = floor[addr].expect("Cannot ADD to empty tile");
                state.held = Some(held.expect("Cannot ADD with empty hands") + other);
            }
            Instruction::Sub(addr) => {
                let addr = addr.get_value(state);
                let other = floor[addr].expect("Cannot SUB from empty tile");
                state.held = Some(held.expect("Cannot SUB with empty hands") - other);
            }
            Instruction::BumpUp(addr) => {
                let addr = addr.get_value(state);
                let other = floor[addr].expect("Cannot BUMP empty tile");
                let res = other + OfficeTile::Number(1);
                state.floor[addr] = Some(res);
                state.held = Some(res);
            }
            Instruction::BumpDown(addr) => {
                let addr = addr.get_value(state);
                let other = floor[addr].expect("Cannot BUMP empty tile");
                let res = other - OfficeTile::Number(1);
                state.floor[addr] = Some(res);
                state.held = Some(res);
            }
            Instruction::CopyFrom(addr) => {
                let addr = addr.get_value(state);
                let tile = floor[addr].expect("Cannot COPYFROM empty tile");
                state.held = Some(tile)
            }
            Instruction::CopyTo(addr) => {
                let held = held.expect("Cannot COPYTO with empty hands");
                let addr = addr.get_value(state);
                state.floor[addr] = Some(held);
            }
            Instruction::Inbox => {
                //TODO: end gracefully here?
                let val = state.inbox.pop_front().expect("Nothing left to INBOX");
                println!("Fetching {:?} from inbox", val);
                println!("Inbox:{:?}", state.inbox);
                state.held = Some(val);
            }
            Instruction::Outbox => {
                let held = held.expect("Cannot OUTBOX with empty hands");
                state.outbox.push_front(held);
                state.held = None;
            }
            Instruction::LabelDef(_)
            | Instruction::Jump(_)
            | Instruction::JumpN(_)
            | Instruction::JumpZ(_) => {}
            Instruction::Comment => {}
            Instruction::Define => {}
        }
    }
}

fn tokenize_hrm(file: File) -> std::io::Result<(Vec<TokenDebug>)> {
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_number = 0;
    {
        //Ensure program starts with the proper header
        let header = "-- HUMAN RESOURCE MACHINE PROGRAM --\n";
        reader.read_line(&mut line)?;
        line_number += 1;
        if line != header {
            eprintln!("Expected {} got {}", header, line);
            panic!("File is not human resource machine file");
        }
        line.clear();
    }

    let mut tokens_vec: Vec<TokenDebug> = Vec::new();
    while reader.read_line(&mut line)? != 0 {
        line_number += 1;
        let tokens = line.split_whitespace();
        for token in tokens {
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
                "COMMENT" => Token::Op(Op::Comment),
                "DEFINE" => Token::Op(Op::Define),
                "LABEL" => Token::Op(Op::Label),
                define if define.ends_with(';') => {
                    /* serialized svg comment, can safely be ignored*/
                    println!("defined comment as {}", define);
                    Token::Op(Op::Svg(String::from(define)))
                }
                label if label.ends_with(':') => {
                    if label.len() == 1 {
                        //invalid - the empty label is not a label
                        panic!("invalid label at line {}", line_number);
                    }
                    let label_name = String::from(label.split_at(label.len() - 1).0);
                    Token::Op(Op::LabelDef(label_name))
                    //Token::Label(Label { name: label_name })
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
                debug_info: DebugInfo { line_number },
            })
        }
        line.clear();
    }
    Ok(tokens_vec)
}

fn tokens_to_instructions(tokens: Vec<TokenDebug>) -> Vec<Instruction> {
    let mut instrs = Vec::new();
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        let debg = token.debug_info;
        let token = token.token;

        match token {
            Token::Op(op) => match op {
                Op::Inbox => instrs.push(Instruction::Inbox),
                Op::Outbox => instrs.push(Instruction::Outbox),
                Op::Comment => instrs.push(Instruction::Comment),
                Op::Define => instrs.push(Instruction::Define),
                Op::CopyFrom | Op::CopyTo | Op::BumpUp | Op::BumpDown | Op::Add | Op::Sub => {
                    let next = &tokens.next().expect("op requires address argument");
                    let next = &next.token;
                    if let Token::Address(addr) = next {
                        match op {
                            Op::CopyFrom => instrs.push(Instruction::CopyFrom(*addr)),
                            Op::CopyTo => instrs.push(Instruction::CopyTo(*addr)),
                            Op::BumpUp => instrs.push(Instruction::BumpUp(*addr)),
                            Op::BumpDown => instrs.push(Instruction::BumpDown(*addr)),
                            Op::Add => instrs.push(Instruction::Add(*addr)),
                            Op::Sub => instrs.push(Instruction::Sub(*addr)),
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
                            Op::Jump => instrs.push(Instruction::Jump(label.clone())),
                            Op::JumpN => instrs.push(Instruction::JumpN(label.clone())),
                            Op::JumpZ => instrs.push(Instruction::JumpZ(label.clone())),
                            _ => panic!("Interpreter error, case not covered"),
                        },
                        _ => panic!(format!("Expected address, found {:?}", next)),
                    }

                }
                Op::LabelDef(label) => instrs.push(Instruction::LabelDef(label)),
                Op::Label => {}
                Op::Svg(_svg) => {}
            },
            Token::Address(_address) => {
                eprintln!("{:?}", debg);
                panic!("Address requires op taking address")
            }
            Token::Label(_label) => {
                eprintln!("{:?}", debg);
                panic!("Label requires op taking label")
            }
        }
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

fn interpret(instructions: &[Instruction], state: &mut OfficeState) {
    let jmp_map = process_labels(&instructions);
    let mut instr_ptr = 0_usize;
    println!("{}", state);
    while instr_ptr < instructions.len() {
        let instruction = &instructions[instr_ptr];
        println!("Executing {:?}", instruction);
        instruction.execute(state);
        if let Some(index) = calc_jump(instruction, &jmp_map, state.held) {
            instr_ptr = index;
        } else {
            instr_ptr += 1;
        }
        println!("{}", state);
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

fn main() -> std::io::Result<()> {
    let file = File::open("example.hrm")?;
    let tokens = tokenize_hrm(file)?;
    //println!("{:?}", tokens);
    let instructions = tokens_to_instructions(tokens);
    println!("{:?}", instructions);

    let inbox: VecDeque<OfficeTile> = vec![
        OfficeTile::from_char('b'),
        OfficeTile::from_char('r'),
        OfficeTile::from_char('a'),
        OfficeTile::from_char('i'),
        OfficeTile::from_char('n'),
        OfficeTile::from_number(0),
    ]
    .into_iter()
    .collect();
    let floor_size = 15;
    let mut floor = Vec::with_capacity(floor_size);
    for _ in 0..floor_size {
        floor.push(None);
    }
    floor[14] = Some(OfficeTile::from_number(0));

    let mut office_state = OfficeState::new_with_inbox_floor(inbox, floor);
    interpret(&instructions, &mut office_state);
    //println!("{}", office_state);
    Ok(())
}
