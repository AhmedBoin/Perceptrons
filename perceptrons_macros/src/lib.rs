use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprBinary};

#[rustfmt::skip]
#[proc_macro]
pub fn mask_tensor(input: TokenStream) -> TokenStream {
    let b = parse_macro_input!(input as ExprBinary);
    let left = &b.left;
    let op = &b.op;
    let right = &b.right;
    quote! {
            tensor!(#left.clone().into_dyn().iter().zip(#right.clone().into_dyn().iter()).map(|(a, b)| (a #op b) as u8 as f32).collect::<Vec<f32>>()).reshape(#left.shape())
        }.into()
}
