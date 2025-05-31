extern crate clap;
use clap::{App, Arg};
use egraphslib::*;
use std::env;
use std::fs::File;
use std::io::Write;

fn generate_rules(vector_width: usize) {
    let rules = rules::rules0(vector_width);
    let mut file = File::create("rules.txt").expect("Unable to create file");
    for rule in rules {
        eprintln!("{:?}", rule);
        writeln!(file, "{:?}", rule).expect("Unable to write to file");
    }
}

fn main() {
    let matches = App::new("Rewriter")
        .arg(
            Arg::with_name("MODE")
                .help("Sets the input expression")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("VECTOR_WIDTH")
                .help("Sets the vector_width")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::with_name("EXPRESSION")
                .help("Sets the input expression")
                .required(false)
                .index(3),
        )
        
        .arg(
            Arg::with_name("RULE_NAME")
                .help("Sets the rule name to use")
                .required(false)
                .index(4),
        ).arg(
            Arg::with_name("TARGET_ECLASS")
                .help("Sets the target eclass on which to apply the rule")
                .required(false)
                .index(5),
        )
        .get_matches();

    
    let mode = matches.value_of("MODE").unwrap();
    let vector_width: usize = matches
    .value_of("VECTOR_WIDTH")
    .unwrap()
    .parse()
    .expect("Number must be a valid usize");


    match mode {
        "rules" => {
            eprintln!("Generating rules for width: {:?}", vector_width);
            generate_rules(vector_width);
        },
        
        "run" => {
            let expr_str = matches.value_of("EXPRESSION").unwrap();
            let timeout = env::var("TIMEOUT")
                .ok()
                .and_then(|t| t.parse::<u64>().ok())
                .unwrap_or(500);

            let prog: egg::RecExpr<egraphslib::veclang::VecLang> = expr_str.parse().unwrap();

            let target_eclass = matches.value_of("TARGET_ECLASS").map(|s| s.parse::<usize>().expect("TARGET_ECLASS must be a valid usize"));

            let rule_name = matches.value_of("RULE_NAME").unwrap();
            let (cost, best);

            
            if let Some(target_eclass) = target_eclass {
                (cost, best) = rules::apply_rule_eclass(&prog, timeout, vector_width, rule_name, target_eclass.into());
            } else {
                eprintln!("TARGET_ECLASS is required for run_one mode");
                return;
            }
           

            println!("{}", best.to_string());
            println!("Cost: {}", cost);
        },
        "get_matches" => {
            let expr_str = matches.value_of("EXPRESSION").unwrap();
            let timeout = env::var("TIMEOUT")
                .ok()
                .and_then(|t| t.parse::<u64>().ok())
                .unwrap_or(500);

            let prog: egg::RecExpr<egraphslib::veclang::VecLang> = expr_str.parse().unwrap();


            let rule_name = matches.value_of("RULE_NAME").unwrap();

            let matches = rules::get_rule_matches(&prog, timeout, vector_width, rule_name);

            println!("{:?}",matches);

       
        },
         "run_batch" => {
            let expr_str = matches.value_of("EXPRESSION").unwrap();
            let timeout = env::var("TIMEOUT")
                .ok()
                .and_then(|t| t.parse::<u64>().ok())
                .unwrap_or(500);

            let prog: egg::RecExpr<egraphslib::veclang::VecLang> = expr_str.parse().unwrap();


            let rule_name: Vec<String> = matches.value_of("RULE_NAME").unwrap().split(',').map(|s| s.to_string()).collect();

            let (cost, best);
            (cost, best) = rules::run(&prog, timeout, vector_width, rule_name);
            println!("{}", best.to_string());
            println!("Cost: {}", cost);

       
        },
        _ => {
            eprintln!("Invalid mode: {}", mode);
        }
    }

    
}
